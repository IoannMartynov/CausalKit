from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Optional lightweight dataclass for standalone usage, but CausalEDA also
# supports the existing causalkit.data.CausalData which uses `cofounders`.
@dataclass
class CausalDataLite:
    df: pd.DataFrame
    treatment: str
    target: str
    confounders: List[str]


def _extract_roles(data_obj: Any) -> Dict[str, Any]:
    """Extract roles from either CausalDataLite, the project's CausalData, or a
    duck-typed object with similar attributes. Supports both `confounders` and
    `cofounders` spellings.
    """
    # Direct dataclass-like attributes
    df = getattr(data_obj, "df")
    treatment_attr = getattr(data_obj, "treatment")
    target_attr = getattr(data_obj, "target")
    # If these are Series (as in causalkit.data.CausalData properties), convert to column names
    if isinstance(treatment_attr, pd.Series):
        treatment = treatment_attr.name
    else:
        treatment = treatment_attr
    if isinstance(target_attr, pd.Series):
        target = target_attr.name
    else:
        target = target_attr


    if hasattr(data_obj, "confounders") and getattr(data_obj, "confounders") is not None:
        confs = list(getattr(data_obj, "confounders"))
    elif hasattr(data_obj, "cofounders") and getattr(data_obj, "cofounders") is not None:
        # causalkit.data.CausalData.cofounders returns a DataFrame or None; if it's a
        # DataFrame, use its columns; if it's a list/iterable, cast to list.
        cofs = getattr(data_obj, "cofounders")
        if isinstance(cofs, pd.DataFrame):
            confs = list(cofs.columns)
        else:
            confs = list(cofs) if cofs is not None else []
    else:
        # Last resort: assume all columns except treatment/target are confounders
        confs = [c for c in df.columns if c not in {treatment, target}]

    return {"df": df, "treatment": treatment, "target": target, "confounders": confs}


class CausalEDA:
    def __init__(self, data: Any, ps_model: Optional[Any] = None, n_splits: int = 5, random_state: int = 42):
        roles = _extract_roles(data)
        self.d = CausalDataLite(df=roles["df"], treatment=roles["treatment"], target=roles["target"], confounders=roles["confounders"])
        self.n_splits = n_splits
        self.random_state = random_state
        self.ps_model = ps_model or LogisticRegression(max_iter=200)

        # minimal preprocessing for mixed types
        X = self.d.df[self.d.confounders]
        num = X.select_dtypes(include=[np.number]).columns.tolist()
        cat = [c for c in X.columns if c not in num]
        # Scale numeric features to improve numerical conditioning
        num_transformer = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])
        self.preproc = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), cat),
            ],
            remainder="drop",
        )
        self.ps_pipe = Pipeline([("prep", self.preproc), ("clf", self.ps_model)])

    # ---------- basics ----------
    def missingness_report(self) -> pd.DataFrame:
        df = self.d.df
        miss = df.isna().mean().rename("missing_rate").to_frame()
        miss["n_missing"] = df.isna().sum()
        return miss.sort_values("missing_rate", ascending=False)

    def data_health_check(self) -> Dict[str, Any]:
        df = self.d.df
        const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        dups = int(df.duplicated().sum())
        return {"constant_columns": const, "n_duplicates": dups, "n_rows": len(df)}

    def summaries(self) -> Dict[str, Any]:
        df, t, y = self.d.df, self.d.treatment, self.d.target
        if not pd.api.types.is_numeric_dtype(df[t]):
            raise ValueError("Treatment must be numeric 0/1 for summaries().")
        treat_rate = df[t].mean()
        grp = df.groupby(t)[y].agg(["count", "mean", "std"])
        # If either group missing, fill with nan-safe
        naive_diff = np.nan
        try:
            naive_diff = grp.loc[1, "mean"] - grp.loc[0, "mean"]
        except Exception:
            pass
        return {"treatment_rate": treat_rate, "outcome_by_treatment": grp, "naive_diff": naive_diff}

    # ---------- propensity & overlap ----------
    def fit_propensity(self) -> np.ndarray:
        df = self.d.df
        X = df[self.d.confounders]
        t = df[self.d.treatment].astype(int).values
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # Suppress spurious RuntimeWarnings from low-level BLAS matmul in sklearn
        import warnings
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ps = cross_val_predict(self.ps_pipe, X, t, cv=cv, method="predict_proba")[:, 1]
        # clip away from 0/1 for stability
        ps = np.clip(ps, 1e-6, 1 - 1e-6)
        self._ps = ps
        return ps

    def treatment_predictability_auc(self, ps: Optional[np.ndarray] = None) -> float:
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
        # AUC against actual treatment
        t = self.d.df[self.d.treatment].astype(int).values
        return float(roc_auc_score(t, ps))

    def positivity_check(self, ps: Optional[np.ndarray] = None, bounds: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
        low, high = bounds
        share_low = float((ps < low).mean())
        share_high = float((ps > high).mean())
        flag = (share_low + share_high) > 0.02  # heuristic
        return {"bounds": bounds, "share_below": share_low, "share_above": share_high, "flag": bool(flag)}

    def plot_ps_overlap(self, ps: Optional[np.ndarray] = None):
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
        df = self.d.df
        t = df[self.d.treatment].astype(int).values
        plt.figure()
        plt.hist(ps[t == 1], bins=30, alpha=0.5, density=True, label="treated")
        plt.hist(ps[t == 0], bins=30, alpha=0.5, density=True, label="control")
        plt.xlabel("Propensity score")
        plt.ylabel("Density")
        plt.legend()
        plt.title("PS overlap")

    # ---------- balance & weights ----------
    @staticmethod
    def _weighted_mean_var(x, w):
        w = w / w.sum()
        m = np.sum(w * x)
        v = np.sum(w * (x - m) ** 2) / (1 - np.sum(w ** 2)) if (1 - np.sum(w ** 2)) > 0 else np.nan
        return m, v

    def _iptw_weights(self, ps: np.ndarray, estimand: str = "ATE") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.d.df[self.d.treatment].astype(int).values
        if estimand.upper() == "ATE":
            w_t = t / ps + (1 - t) / (1 - ps)
            w_1 = 1 / ps
            w_0 = 1 / (1 - ps)
        elif estimand.upper() == "ATT":
            w_t = np.where(t == 1, 1.0, ps / (1 - ps))
            w_1 = np.ones_like(ps)
            w_0 = ps / (1 - ps)
        else:
            raise ValueError("estimand must be 'ATE' or 'ATT'")
        return w_t, w_1, w_0

    def balance_table(self, ps: Optional[np.ndarray] = None, estimand: str = "ATE") -> pd.DataFrame:
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
        df = self.d.df
        X = df[self.d.confounders]
        t = df[self.d.treatment].astype(int).values
        # convert cats to dummies for SMDs
        X_num = pd.get_dummies(X, drop_first=False)
        _, w1, w0 = self._iptw_weights(ps, estimand)
        rows = []
        for col in X_num.columns:
            x = X_num[col].values.astype(float)
            # unweighted
            m1 = x[t == 1].mean()
            v1 = x[t == 1].var(ddof=1)
            m0 = x[t == 0].mean()
            v0 = x[t == 0].var(ddof=1)
            smd = (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0
            # weighted (IPTW)
            m1w, v1w = self._weighted_mean_var(x[t == 1], w1[t == 1])
            m0w, v0w = self._weighted_mean_var(x[t == 0], w0[t == 0])
            smd_w = (m1w - m0w) / np.sqrt((v1w + v0w) / 2) if (v1w + v0w) > 0 else 0.0
            rows.append({"covariate": col, "SMD_unweighted": smd, "SMD_weighted": smd_w})
        tab = pd.DataFrame(rows).sort_values("SMD_unweighted", key=lambda s: s.abs(), ascending=False)
        tab["flag_unw"] = tab["SMD_unweighted"].abs() > 0.1
        tab["flag_w"] = tab["SMD_weighted"].abs() > 0.1
        return tab

    def love_plot(self, balance_df: pd.DataFrame, top_n: int = 25):
        d = balance_df.copy().head(top_n)
        plt.figure()
        y = np.arange(len(d))[::-1]
        plt.scatter(d["SMD_unweighted"], y, label="pre")
        plt.scatter(d["SMD_weighted"], y, marker="x", label="post")
        for i, name in enumerate(d["covariate"]):
            plt.text(-0.02, y[i], str(name), va="center", ha="right")
        plt.axvline(0.1, linestyle="--")
        plt.axvline(-0.1, linestyle="--")
        plt.yticks([])
        plt.xlabel("Standardized mean difference")
        plt.legend()
        plt.title("Love plot")

    def weight_diagnostics(self, ps: Optional[np.ndarray] = None, estimand: str = "ATE") -> Dict[str, Any]:
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
        t = self.d.df[self.d.treatment].astype(int).values
        w_all, w1, w0 = self._iptw_weights(ps, estimand)

        def ess(w):
            s = w.sum()
            return (s * s) / np.sum(w * w)

        return {
            "ESS_all": float(ess(w_all)),
            "ESS_treated": float(ess(w1[t == 1])),
            "ESS_control": float(ess(w0[t == 0])),
            "w_all_quantiles": np.quantile(w_all, [0.5, 0.9, 0.95, 0.99, 1.0]).tolist(),
        }

    # ---------- one-shot driver ----------
    def design_report(self) -> Dict[str, Any]:
        ps = self.fit_propensity()
        return {
            "health": self.data_health_check(),
            "missing": self.missingness_report(),
            "summaries": self.summaries(),
            "treat_auc": self.treatment_predictability_auc(ps),
            "positivity": self.positivity_check(ps),
            "balance": self.balance_table(ps),
            "weights": self.weight_diagnostics(ps),
        }
