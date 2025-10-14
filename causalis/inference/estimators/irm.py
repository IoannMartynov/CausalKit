r"""
DML IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE scores.
This is a lightweight clone of DoubleML's IRM tailored for CausalData input.

software{DoubleML,
  title = {{DoubleML} -- Double Machine Learning in Python},
  author = {Bach, Philipp and Chernozhukov, Victor and Klaassen, Sven and Kurz, Malte S. and Spindler, Martin},
  year = {2024},
  version = {latest},
  url = {https://github.com/DoubleML/doubleml-for-py},
  note = {BSD-3-Clause License. Documentation: \url{https://docs.doubleml.org/stable/index.html}}
}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sklearn.base import is_classifier, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm

from causalis.data.causaldata import CausalData


def _compute_sensitivity_bias(sigma2: np.ndarray | float,
                               nu2: np.ndarray | float,
                               psi_sigma2: np.ndarray,
                               psi_nu2: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute (max) sensitivity bias and its influence function, following DoubleML.
    
    Parameters
    ----------
    sigma2 : float or array-like
        Variance of residuals of the structural equation for Y.
    nu2 : float or array-like
        Sensitivity parameter related to the Riesz representer and IPW.
    psi_sigma2 : np.ndarray
        Influence function terms for sigma2.
    psi_nu2 : np.ndarray
        Influence function terms for nu2.
    
    Returns
    -------
    max_bias : float
        The maximum bias under the given sensitivity parameters.
    psi_max_bias : np.ndarray
        The influence function for max_bias used for standard error of bias.
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    nu2_f = float(np.asarray(nu2).reshape(()))
    max_bias = float(np.sqrt(max(sigma2_f * nu2_f, 0.0)))
    # Avoid division by zero
    denom = 2.0 * (max_bias if max_bias > 0 else 1.0)
    psi_max_bias = (sigma2_f * psi_nu2 + nu2_f * psi_sigma2) / denom
    return max_bias, psi_max_bias


def _combine_nu2(m_alpha: np.ndarray, rr: np.ndarray, cf_y: float, cf_d: float, rho: float) -> tuple[float, np.ndarray]:
    """Combine sensitivity levers into nu2 via per-unit quadratic form.

    nu2_i = (sqrt(2*m_alpha_i))^2 * cf_y + (|rr_i|)^2 * cf_d + 2*rho*sqrt(cf_y*cf_d)*|rr_i|*sqrt(2*m_alpha_i)

    Returns
    -------
    nu2 : float
        Mean of per-unit nu2_i.
    psi_nu2 : np.ndarray
        Influence-function-style centered terms: nu2_i - nu2.
    """
    cf_y = float(cf_y)
    cf_d = float(cf_d)
    # clip rho to [-1,1]
    rho = float(np.clip(rho, -1.0, 1.0))
    if cf_y < 0 or cf_d < 0:
        raise ValueError("cf_y and cf_d must be >= 0.")
    a = np.sqrt(2.0 * np.maximum(np.asarray(m_alpha, dtype=float), 0.0))
    b = np.abs(np.asarray(rr, dtype=float))
    base = (a * a) * cf_y + (b * b) * cf_d + 2.0 * rho * np.sqrt(cf_y * cf_d) * a * b
    nu2 = float(np.mean(base))
    psi_nu2 = base - nu2
    return nu2, psi_nu2


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
        self.folds_: Optional[np.ndarray] = None
        # Sensitivity & data cache
        self._y: Optional[np.ndarray] = None
        self._d: Optional[np.ndarray] = None
        self.sensitivity_summary: Optional[str] = None

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
        # Cache for sensitivity analysis
        self._y = y.copy()
        self._d = d.copy()
        n = X.shape[0]

        # Enforce valid propensity model: must expose predict_proba when classifier
        if is_classifier(self.ml_m) and not hasattr(self.ml_m, "predict_proba"):
            raise ValueError("ml_m must support predict_proba() to produce valid propensity probabilities.")
        # For binary outcomes, require probabilistic outcome models when using classifiers
        if y_is_binary and is_classifier(self.ml_g) and not hasattr(self.ml_g, "predict_proba"):
            raise ValueError("Binary outcome: ml_g is a classifier but does not expose predict_proba(). Use a probabilistic classifier or calibrate it.")

        if self.n_rep != 1:
            raise NotImplementedError("IRM currently supports n_rep=1 only.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.trimming_rule not in {"truncate"}:
            raise ValueError("Only trimming_rule='truncate' is supported")

        g0_hat = np.full(n, np.nan, dtype=float)
        g1_hat = np.full(n, np.nan, dtype=float)
        m_hat = np.full(n, np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, d)):
            folds[test_idx] = i
            # Outcome models trained on respective treatment groups in the train fold
            X_tr, y_tr, d_tr = X[train_idx], y[train_idx], d[train_idx]
            X_te = X[test_idx]

            # g0
            model_g0 = clone(self.ml_g)
            mask0 = (d_tr == 0)
            if not np.any(mask0):
                # Fallback: if no control in train, fit on all train
                warnings.warn("IRM: A CV fold had no controls in the training split; fitting g0 on all train data for this fold.", RuntimeWarning, stacklevel=2)
                X_g0, y_g0 = X_tr, y_tr
            else:
                X_g0, y_g0 = X_tr[mask0], y_tr[mask0]
            model_g0.fit(X_g0, y_g0)
            if y_is_binary and is_classifier(model_g0) and hasattr(model_g0, "predict_proba"):
                pred_g0 = model_g0.predict_proba(X_te)
                pred_g0 = pred_g0[:, 1] if pred_g0.ndim == 2 else pred_g0.ravel()
            else:
                pred_g0 = model_g0.predict(X_te)
            pred_g0 = np.asarray(pred_g0, dtype=float).ravel()
            if y_is_binary:
                pred_g0 = np.clip(pred_g0, 1e-12, 1 - 1e-12)
            g0_hat[test_idx] = pred_g0

            # g1
            model_g1 = clone(self.ml_g)
            mask1 = (d_tr == 1)
            if not np.any(mask1):
                warnings.warn("IRM: A CV fold had no treated units in the training split; fitting g1 on all train data for this fold.", RuntimeWarning, stacklevel=2)
                X_g1, y_g1 = X_tr, y_tr
            else:
                X_g1, y_g1 = X_tr[mask1], y_tr[mask1]
            model_g1.fit(X_g1, y_g1)
            if y_is_binary and is_classifier(model_g1) and hasattr(model_g1, "predict_proba"):
                pred_g1 = model_g1.predict_proba(X_te)
                pred_g1 = pred_g1[:, 1] if pred_g1.ndim == 2 else pred_g1.ravel()
            else:
                pred_g1 = model_g1.predict(X_te)
            pred_g1 = np.asarray(pred_g1, dtype=float).ravel()
            if y_is_binary:
                pred_g1 = np.clip(pred_g1, 1e-12, 1 - 1e-12)
            g1_hat[test_idx] = pred_g1

            # m
            model_m = clone(self.ml_m)
            model_m.fit(X_tr, d_tr)
            m_pred = _predict_prob_or_value(model_m, X_te)
            m_hat[test_idx] = m_pred

        # Trimming/clipping propensity
        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g0_hat)) or np.any(np.isnan(g1_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        m_hat = _clip_propensity(m_hat, self.trimming_threshold)
        self.folds_ = folds

        # Score elements
        u0 = y - g0_hat
        u1 = y - g1_hat
        h1, h0 = self._normalize_ipw_terms(d, m_hat)

        # weights
        w, w_bar = self._get_weights(n, m_hat, d)

        # psi elements
        if self.score == "ATTE" and self.normalize_ipw:
            # Rescale w_bar per leg to preserve ATT score identities under IPW normalization
            s1 = float(np.mean(d / m_hat))
            s0 = float(np.mean((1 - d) / (1 - m_hat)))
            s1 = s1 if s1 != 0 else 1.0
            s0 = s0 if s0 != 0 else 1.0
            # Note: h1, h0 may already be normalized to mean 1; scaling w_bar per leg preserves products
            psi_b = w * (g1_hat - g0_hat) + (u1 * h1) * (w_bar * s1) - (u0 * h0) * (w_bar * s0)
        else:
            psi_b = w * (g1_hat - g0_hat) + w_bar * (u1 * h1 - u0 * h0)
        w_mean = float(np.mean(w))
        psi_a = -w / (w_mean if w_mean != 0.0 else 1.0)  # ensures E[psi_a] = -1 with zero-guard

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

    # --------- Sensitivity (DoubleML-style) ---------
    def _sensitivity_element_est(self) -> dict:
        """Compute elements needed for sensitivity bias bounds.
        Mirrors DoubleMLIRM._sensitivity_element_est using fitted nuisances.
        Requires fit() to have been called.
        """
        if any(getattr(self, attr) is None for attr in ["g0_hat_", "g1_hat_", "m_hat_", "coef_"]):
            raise RuntimeError("IRM model must be fitted before sensitivity analysis.")
        y = self._y
        d = self._d
        if y is None or d is None:
            # fallback to current data
            df = self.data.get_df()
            y = df[self.data.target.name].to_numpy(dtype=float)
            d = df[self.data.treatment.name].to_numpy(dtype=int)
        m_hat = np.asarray(self.m_hat_, dtype=float)
        g0 = np.asarray(self.g0_hat_, dtype=float)
        g1 = np.asarray(self.g1_hat_, dtype=float)

        # Adjusted propensity (we use clipped m_hat)
        m_hat_adj = m_hat
        # weights
        w, w_bar = self._get_weights(n=len(y), m_hat_adj=m_hat_adj, d=d)

        # sigma^2
        sigma2_score_element = np.square(y - d * g1 - (1.0 - d) * g0)
        sigma2 = float(np.mean(sigma2_score_element))
        psi_sigma2 = sigma2_score_element - sigma2

        # Riesz representer and m_alpha
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_m = 1.0 / m_hat_adj
            inv_1m = 1.0 / (1.0 - m_hat_adj)
        m_alpha = w * w_bar * (inv_m + inv_1m)
        rr = w_bar * (d * inv_m - (1.0 - d) * inv_1m)

        nu2_score_element = 2.0 * m_alpha - np.square(rr)
        nu2 = float(np.mean(nu2_score_element))
        psi_nu2 = nu2_score_element - nu2

        return {
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": psi_sigma2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
            "m_alpha": m_alpha,
        }

    def sensitivity_analysis(self, cf_y: float, cf_d: float, rho: float = 1.0, level: float = 0.95) -> "IRM":
        """Compute a simple sensitivity summary and store it as `sensitivity_summary`.
        
        Parameters
        ----------
        cf_y : float
            Sensitivity parameter for outcome equation.
        cf_d : float
            Sensitivity parameter for treatment equation.
        rho : float, default 1.0
            Correlation between unobserved components (display only here).
        level : float, default 0.95
            Confidence level for CI bounds display.
        """
        # Validate inputs
        if not (0.0 < float(level) < 1.0):
            raise ValueError("level must be in (0,1).")
        if cf_y < 0 or cf_d < 0:
            raise ValueError("cf_y and cf_d must be >= 0.")
        rho = float(np.clip(rho, -1.0, 1.0))

        # Gather elements
        elems = self._sensitivity_element_est()
        sigma2 = elems["sigma2"]
        psi_sigma2 = elems["psi_sigma2"]
        m_alpha = elems.get("m_alpha")
        rr = elems.get("riesz_rep")

        # Build nu2 from components and sensitivity levers
        nu2, psi_nu2 = _combine_nu2(m_alpha, rr, cf_y, cf_d, rho)

        max_bias, psi_max_bias = _compute_sensitivity_bias(sigma2, nu2, psi_sigma2, psi_nu2)

        theta_hat = float(self.coef_[0])
        se = float(self.se_[0])
        z = norm.ppf(0.5 + level / 2.0)
        ci_low = theta_hat - z * se
        ci_high = theta_hat + z * se

        theta_lower = theta_hat - max_bias
        theta_upper = theta_hat + max_bias

        # Format summary text similar to DoubleML
        lines = []
        lines.append("================== Sensitivity Analysis ==================")
        lines.append("")
        lines.append("------------------ Scenario          ------------------")
        lines.append(f"Significance Level: level={level}")
        lines.append(f"Sensitivity parameters: cf_y={cf_y}; cf_d={cf_d}, rho={rho}")
        lines.append("")
        lines.append("------------------ Bounds with CI    ------------------")
        header = f"{'':>6} {'CI lower':>11} {'theta lower':>12} {'theta':>15} {'theta upper':>12} {'CI upper':>13}"
        lines.append(header)
        row_name = self.data.treatment.name
        lines.append(f"{row_name:>6} {ci_low:11.6f} {theta_lower:12.6f} {theta_hat:15.6f} {theta_upper:12.6f} {ci_high:13.6f}")
        lines.append("")
        lines.append("------------------ Robustness (SNR proxy) -------------")
        rob_header = f"{'':>6} {'H_0':>6} {'SNR proxy (%)':>15} {'adj (%)':>8}"
        lines.append(rob_header)
        snr = abs(theta_hat) / (se + 1e-12)
        rv = min(100.0, float(100.0 * (1.0 / (1.0 + snr))))
        rva = max(0.0, rv - 5.0)
        lines.append(f"{row_name:>6} {0.0:6.1f} {rv:15.6f} {rva:8.6f}")

        self.sensitivity_summary = "\n".join(lines)
        return self

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
