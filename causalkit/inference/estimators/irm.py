"""
Internal DoubleML-style IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE scores.
This is a lightweight clone of DoubleML's IRM tailored for CausalData input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sklearn.base import is_classifier, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm

from causalkit.data.causaldata import CausalData


def _is_binary(values: np.ndarray) -> bool:
    uniq = np.unique(values)
    return np.array_equal(np.sort(uniq), np.array([0, 1])) or np.array_equal(np.sort(uniq), np.array([0.0, 1.0]))


def _predict_prob_or_value(model, X: np.ndarray) -> np.ndarray:
    if is_classifier(model) and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            return np.clip(proba.ravel(), 1e-12, 1 - 1e-12)
        return np.clip(proba[:, 1], 1e-12, 1 - 1e-12)
    else:
        preds = model.predict(X)
        return np.asarray(preds, dtype=float).ravel()


def _clip_propensity(p: np.ndarray, thr: float) -> np.ndarray:
    thr = float(thr)
    return np.clip(p, thr, 1.0 - thr)


@dataclass
class IRMResults:
    coef: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    pval: np.ndarray
    confint: np.ndarray
    summary: pd.DataFrame


class IRM:
    """Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

    Parameters
    ----------
    data : CausalData
        Data container with outcome, binary treatment (0/1), and confounders.
    ml_g : estimator
        Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
    ml_m : classifier
        Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
    n_folds : int, default 5
        Number of cross-fitting folds.
    n_rep : int, default 1
        Number of repetitions of sample splitting. Currently only 1 is supported.
    score : {"ATE","ATTE"}, default "ATE"
        Target estimand.
    normalize_ipw : bool, default False
        Whether to normalize IPW terms within the score.
    trimming_rule : {"truncate"}, default "truncate"
        Trimming approach for propensity scores.
    trimming_threshold : float, default 1e-2
        Threshold for trimming if rule is "truncate".
    weights : Optional[np.ndarray or Dict], default None
        Optional weights. If array of shape (n,), used as ATE weights. For ATTE, computed internally.
    random_state : Optional[int], default None
        Random seed for fold creation.
    """

    def __init__(
        self,
        data: CausalData,
        ml_g: Any,
        ml_m: Any,
        *,
        n_folds: int = 5,
        n_rep: int = 1,
        score: str = "ATE",
        normalize_ipw: bool = False,
        trimming_rule: str = "truncate",
        trimming_threshold: float = 1e-2,
        weights: Optional[np.ndarray | Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.score = str(score).upper()
        self.normalize_ipw = bool(normalize_ipw)
        self.trimming_rule = str(trimming_rule)
        self.trimming_threshold = float(trimming_threshold)
        self.weights = weights
        self.random_state = random_state

        # Placeholders after fit
        self.g0_hat_: Optional[np.ndarray] = None
        self.g1_hat_: Optional[np.ndarray] = None
        self.m_hat_: Optional[np.ndarray] = None
        self.psi_a_: Optional[np.ndarray] = None
        self.psi_b_: Optional[np.ndarray] = None
        self.psi_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.se_: Optional[np.ndarray] = None
        self.t_stat_: Optional[np.ndarray] = None
        self.pval_: Optional[np.ndarray] = None
        self.confint_: Optional[np.ndarray] = None
        self.summary_: Optional[pd.DataFrame] = None

    # --------- Helpers ---------
    def _check_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = self.data.get_df().copy()
        y = df[self.data.target.name].to_numpy(dtype=float)
        d = df[self.data.treatment.name].to_numpy()
        # Ensure binary 0/1
        if df[self.data.treatment.name].dtype == bool:
            d = d.astype(int)
        if not _is_binary(d):
            raise ValueError("Treatment must be binary 0/1 or boolean.")
        d = d.astype(int)

        x_cols = list(self.data.confounders)
        if len(x_cols) == 0:
            raise ValueError("CausalData must include non-empty confounders.")
        X = df[x_cols].to_numpy(dtype=float)

        y_is_binary = _is_binary(y)
        return X, y, d, y_is_binary

    def _get_weights(self, n: int, m_hat_adj: Optional[np.ndarray], d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Standard ATE
        if self.score == "ATE":
            if self.weights is None:
                w = np.ones(n, dtype=float)
            elif isinstance(self.weights, np.ndarray):
                if self.weights.shape[0] != n:
                    raise ValueError("weights array must have shape (n,)")
                w = np.asarray(self.weights, dtype=float)
            elif isinstance(self.weights, dict):
                w = np.asarray(self.weights.get("weights"), dtype=float)
                if w.shape[0] != n:
                    raise ValueError("weights['weights'] must have shape (n,)")
            else:
                raise TypeError("weights must be None, np.ndarray, or dict")
            w_bar = w
            if isinstance(self.weights, dict) and "weights_bar" in self.weights:
                w_bar = np.asarray(self.weights["weights_bar"], dtype=float)
                if w_bar.ndim == 2:
                    # choose first repetition for now
                    w_bar = w_bar[:, 0]
            return w, w_bar
        # ATTE requires m_hat
        elif self.score == "ATTE":
            if m_hat_adj is None:
                raise ValueError("m_hat required for ATTE weights computation")
            base_w = np.ones(n, dtype=float)
            subgroup = base_w * d
            subgroup_prob = float(np.mean(subgroup)) if np.mean(subgroup) > 0 else 1.0
            w = subgroup / subgroup_prob
            w_bar = (m_hat_adj * base_w) / subgroup_prob
            return w, w_bar
        else:
            raise ValueError("score must be 'ATE' or 'ATTE'")

    def _normalize_ipw_terms(self, d: np.ndarray, m_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Compute IPW terms and optionally normalize to mean 1
        h1 = d / m_hat
        h0 = (1 - d) / (1 - m_hat)
        if self.normalize_ipw:
            h1_mean = np.mean(h1)
            h0_mean = np.mean(h0)
            # Avoid division by zero
            h1 = h1 / (h1_mean if h1_mean != 0 else 1.0)
            h0 = h0 / (h0_mean if h0_mean != 0 else 1.0)
        return h1, h0

    # --------- API ---------
    def fit(self) -> "IRM":
        X, y, d, y_is_binary = self._check_data()
        n = X.shape[0]

        if self.n_rep != 1:
            raise NotImplementedError("IRM currently supports n_rep=1 only.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.trimming_rule not in {"truncate"}:
            raise ValueError("Only trimming_rule='truncate' is supported")

        g0_hat = np.full(n, np.nan, dtype=float)
        g1_hat = np.full(n, np.nan, dtype=float)
        m_hat = np.full(n, np.nan, dtype=float)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in skf.split(X, d):
            # Outcome models trained on respective treatment groups in the train fold
            X_tr, y_tr, d_tr = X[train_idx], y[train_idx], d[train_idx]
            X_te = X[test_idx]

            # g0
            model_g0 = clone(self.ml_g)
            mask0 = (d_tr == 0)
            if not np.any(mask0):
                # Fallback: if no control in train, fit on all train
                X_g0, y_g0 = X_tr, y_tr
            else:
                X_g0, y_g0 = X_tr[mask0], y_tr[mask0]
            model_g0.fit(X_g0, y_g0)
            if y_is_binary and is_classifier(model_g0) and hasattr(model_g0, "predict_proba"):
                pred_g0 = model_g0.predict_proba(X_te)
                pred_g0 = pred_g0[:, 1] if pred_g0.ndim == 2 else pred_g0.ravel()
            else:
                pred_g0 = model_g0.predict(X_te)
            g0_hat[test_idx] = np.asarray(pred_g0, dtype=float).ravel()

            # g1
            model_g1 = clone(self.ml_g)
            mask1 = (d_tr == 1)
            if not np.any(mask1):
                X_g1, y_g1 = X_tr, y_tr
            else:
                X_g1, y_g1 = X_tr[mask1], y_tr[mask1]
            model_g1.fit(X_g1, y_g1)
            if y_is_binary and is_classifier(model_g1) and hasattr(model_g1, "predict_proba"):
                pred_g1 = model_g1.predict_proba(X_te)
                pred_g1 = pred_g1[:, 1] if pred_g1.ndim == 2 else pred_g1.ravel()
            else:
                pred_g1 = model_g1.predict(X_te)
            g1_hat[test_idx] = np.asarray(pred_g1, dtype=float).ravel()

            # m
            model_m = clone(self.ml_m)
            model_m.fit(X_tr, d_tr)
            m_pred = _predict_prob_or_value(model_m, X_te)
            m_hat[test_idx] = m_pred

        # Trimming/clipping propensity
        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g0_hat)) or np.any(np.isnan(g1_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        m_hat = _clip_propensity(m_hat, self.trimming_threshold)

        # Score elements
        u0 = y - g0_hat
        u1 = y - g1_hat
        h1, h0 = self._normalize_ipw_terms(d, m_hat)

        # weights
        w, w_bar = self._get_weights(n, m_hat, d)

        # psi elements
        psi_b = w * (g1_hat - g0_hat) + w_bar * (u1 * h1 - u0 * h0)
        psi_a = -w / np.mean(w)  # ensures E[psi_a] ≈ -1

        theta_hat = float(np.mean(psi_b))  # since E[psi_a] = -1
        psi = psi_b + psi_a * theta_hat
        var = float(np.var(psi, ddof=1)) / n
        se = float(np.sqrt(max(var, 0.0)))

        # Summary stats (single-parameter model)
        t_stat = theta_hat / se if se > 0 else np.nan
        pval = 2 * (1 - norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
        ci_low, ci_high = theta_hat - norm.ppf(0.975) * se, theta_hat + norm.ppf(0.975) * se

        self.g0_hat_ = g0_hat
        self.g1_hat_ = g1_hat
        self.m_hat_ = m_hat
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b
        self.psi_ = psi
        self.coef_ = np.array([theta_hat])
        self.se_ = np.array([se])
        self.t_stat_ = np.array([t_stat])
        self.pval_ = np.array([pval])
        self.confint_ = np.array([[ci_low, ci_high]])

        self.summary_ = pd.DataFrame(
            {
                "coef": self.coef_,
                "std err": self.se_,
                "t": self.t_stat_,
                "P>|t|": self.pval_,
                "2.5 %": self.confint_[:, 0],
                "97.5 %": self.confint_[:, 1],
            },
            index=[self.data.treatment.name],
        )

        return self

    # Convenience properties similar to DoubleML
    @property
    def coef(self) -> np.ndarray:
        check_is_fitted(self, attributes=["coef_"])
        return self.coef_

    @property
    def se(self) -> np.ndarray:
        check_is_fitted(self, attributes=["se_"])
        return self.se_

    @property
    def pvalues(self) -> np.ndarray:
        check_is_fitted(self, attributes=["pval_"])
        return self.pval_

    @property
    def summary(self) -> pd.DataFrame:
        check_is_fitted(self, attributes=["summary_"])
        return self.summary_

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        check_is_fitted(self, attributes=["coef_", "se_"])
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0,1)")
        z = norm.ppf(0.5 + level / 2.0)
        ci_low = self.coef_[0] - z * self.se_[0]
        ci_high = self.coef_[0] + z * self.se_[0]
        return pd.DataFrame(
            {f"{(1-level)/2*100:.1f} %": [ci_low], f"{(0.5+level/2)*100:.1f} %": [ci_high]},
            index=[self.data.treatment.name],
        )
