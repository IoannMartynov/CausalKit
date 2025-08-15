"""EDA utilities for causal analysis (propensity, overlap, balance, weights).

This module provides a lightweight CausalEDA class to quickly assess whether a
binary treatment problem is suitable for causal effect estimation. The outputs
focus on interpretability: treatment predictability, overlap/positivity,
covariate balance before/after weighting, and basic data health.

What the main outputs mean
- missingness_report(): DataFrame with per-column missing_rate (fraction of NaNs)
  and n_missing. High missingness may require imputation or dropping columns.
- data_health_check(): Dict with constant_columns (uninformative features),
  n_duplicates (potential data leakage or repetition), and n_rows.
- summaries(): Dict with treatment_rate (share of treated),
  outcome_by_treatment (count/mean/std within treatment groups), and
  naive_diff (treated mean minus control mean; a biased estimate if confounding exists).
- fit_propensity(): Numpy array of cross-validated propensity scores P(T=1|X).
- treatment_predictability_auc(): Float AUC of treatment vs. propensity score.
  Higher AUC implies treatment is predictable from X (more confounding risk).
- positivity_check(): Dict with bounds, share_below, share_above, and flag.
  It reports what share of units have PS outside [low, high]; a large share
  signals poor overlap (violated positivity).
- plot_ps_overlap(): Overlaid histograms of PS for treated vs control.
- balance_table(): DataFrame with standardized mean differences (SMD) for each
  covariate before and after IPTW weighting (SMD_unweighted, SMD_weighted) and
  boolean flags if |SMD|>0.1.
- love_plot(): Visual comparison of SMDs pre/post weighting for top covariates.
- weight_diagnostics(): Dict including effective sample sizes (ESS_all,
  ESS_treated, ESS_control) and quantiles of all IPTW weights (w_all_quantiles).
- design_report(): One-shot dict aggregating all of the above, convenient for
  quick inspection and logging.

Note: The class accepts either the project’s CausalData object (duck-typed) or a
CausalDataLite with explicit fields.
"""
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
    """A minimal container for dataset roles used by CausalEDA.

    Attributes
    - df: The full pandas DataFrame containing treatment, outcome and covariates.
    - treatment: Column name of the binary treatment indicator (0/1).
    - target: Column name of the outcome variable.
    - confounders: List of covariate column names used to model treatment.
    """
    df: pd.DataFrame
    treatment: str
    target: str
    confounders: List[str]


def _extract_roles(data_obj: Any) -> Dict[str, Any]:
    """Extract dataset roles from various supported data containers.

    Accepts:
    - CausalDataLite
    - Project's CausalData (duck-typed: attributes df, treatment, outcome,
      and either confounders or cofounders)
    - Any object exposing the same attributes/properties

    Returns a dict with keys: df, treatment, outcome, confounders.
    If both confounders/cofounders are absent, it assumes all columns except
    treatment/outcome are confounders.
    """
    # Direct dataclass-like attributes
    df = getattr(data_obj, "df")
    treatment_attr = getattr(data_obj, "treatment")
    # Support both 'outcome' (project's CausalData) and 'target' (CausalDataLite)
    target_attr = getattr(data_obj, "outcome", None)
    if target_attr is None:
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
        # Last resort: assume all columns except treatment/outcome are confounders
        confs = [c for c in df.columns if c not in {treatment, target}]

    return {"df": df, "treatment": treatment, "outcome": target, "confounders": confs}


class CausalEDA:
    """Exploratory diagnostics for causal designs with binary treatment.

    The class exposes methods to:
    - Check data health and missingness.
    - Summarize outcome by treatment and naive mean difference.
    - Estimate cross-validated propensity scores and assess treatment
      predictability (AUC) and positivity/overlap.
    - Inspect covariate balance via standardized mean differences (SMD)
      before/after IPTW weighting; visualize with a love plot.
    - Inspect weight distributions and effective sample size (ESS).
    - Produce a one-shot design_report() that aggregates the above.
    """
    def __init__(self, data: Any, ps_model: Optional[Any] = None, n_splits: int = 5, random_state: int = 42):
        roles = _extract_roles(data)
        self.d = CausalDataLite(df=roles["df"], treatment=roles["treatment"], target=roles["outcome"], confounders=roles["confounders"])
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
        """Report per-column missingness.

        Returns a DataFrame indexed by column with:
        - missing_rate: fraction of NaNs per column
        - n_missing: count of NaNs per column
        Also includes a 'total_missing' attribute for accessing the total missing count.
        Sorted by missing_rate descending.
        """
        df = self.d.df
        miss = df.isna().mean().rename("missing_rate").to_frame()
        miss["n_missing"] = df.isna().sum()
        result = miss.sort_values("missing_rate", ascending=False)
        
        # Add total_missing as an attribute that can be accessed with bracket notation
        total_missing = int(df.isna().sum().sum())
        result.__class__ = type('MissingnessDataFrame', (pd.DataFrame,), {
            '__getitem__': lambda self, key: total_missing if key == 'total_missing' else pd.DataFrame.__getitem__(self, key)
        })
        
        return result

    def data_health_check(self) -> Dict[str, Any]:
        """Basic data health indicators.

        Returns a dict with:
        - constant_columns: list of columns with <=1 unique value (uninformative)
        - n_duplicates: number of duplicated rows
        - n_rows: total number of rows
        """
        df = self.d.df
        const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        dups = int(df.duplicated().sum())
        return {"constant_columns": const, "n_duplicates": dups, "n_rows": len(df)}

    def summaries(self) -> Dict[str, Any]:
        """Outcome summaries by treatment.

        Returns a dict with:
        - treatment_rate: share of treated units (mean of 0/1 treatment)
        - outcome_by_treatment: pandas DataFrame with count/mean/std of outcome per treatment group
        - naive_diff: treated mean minus control mean (unadjusted, potentially biased)
        """
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
        """Estimate cross-validated propensity scores P(T=1|X).

        Uses a preprocessing+logistic regression pipeline with stratified K-fold
        cross_val_predict to generate out-of-fold probabilities. Returns a
        numpy array of shape (n_rows,) with values clipped to (1e-6, 1-1e-6).
        """
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
        """Compute AUC of treatment assignment vs. estimated propensity score.

        Interpretation: Higher AUC means treatment is more predictable from X,
        indicating stronger systematic differences between groups (potential
        confounding). Values near 0.5 suggest random-like assignment.
        """
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
        # AUC against actual treatment
        t = self.d.df[self.d.treatment].astype(int).values
        return float(roc_auc_score(t, ps))

    def positivity_check(self, ps: Optional[np.ndarray] = None, bounds: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
        """Check overlap/positivity based on propensity score thresholds.

        Returns a dict with:
        - bounds: (low, high) thresholds used
        - share_below: fraction with PS < low
        - share_above: fraction with PS > high
        - flag: heuristic boolean True if the tails collectively exceed ~2%
        """
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
        """Plot overlaid histograms of propensity scores for treated vs control.

        Useful to visually assess group overlap. Does not return data; it draws
        on the current matplotlib figure.
        """
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
        """Compute IPTW weights for a given estimand.

        Returns (w_t, w_1, w_0):
        - w_t: per-row overall weight for the estimand
        - w_1: weights to reweight the treated group
        - w_0: weights to reweight the control group
        """
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
        """Compute standardized mean differences (SMD) pre/post IPTW.

        Returns a DataFrame with columns:
        - covariate: (possibly one-hot) covariate name
        - SMD_unweighted: standardized mean difference before weighting
        - SMD_weighted: standardized mean difference after IPTW
        - flag_unw: True if |SMD_unweighted| > 0.1
        - flag_w: True if |SMD_weighted| > 0.1
        Lower |SMD| indicates better balance between treated and control.
        """
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
        """Create a love plot comparing pre/post weighting SMDs.

        Parameters
        - balance_df: output of balance_table()
        - top_n: show up to this many covariates (sorted by |SMD_unweighted|)
        """
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
        """Summary stats for IPTW weights and effective sample size (ESS).

        Returns a dict with:
        - ESS_all: effective sample size using overall weights
        - ESS_treated: ESS within the treated group
        - ESS_control: ESS within the control group
        - w_all_quantiles: list with [50%, 90%, 95%, 99%, 100%] quantiles of overall weights
        Larger tails imply a few samples dominate, which can inflate variance.
        """
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

    def plot_target_by_treatment(self,
                                 treatment: Optional[str] = None,
                                 target: Optional[str] = None,
                                 bins: int = 30,
                                 density: bool = True,
                                 alpha: float = 0.5,
                                 figsize: Tuple[float, float] = (7, 4),
                                 sharex: bool = True) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot the distribution of the outcome for every treatment on one plot,
        and also produce a boxplot by treatment to visualize outliers.

        Parameters
        ----------
        treatment : Optional[str]
            Treatment column name. Defaults to the treatment stored in the CausalEDA data.
        target : Optional[str]
            Target/outcome column name. Defaults to the outcome stored in the CausalEDA data.
        bins : int
            Number of bins for histograms when the outcome is numeric.
        density : bool
            Whether to normalize histograms to form a density.
        alpha : float
            Transparency for overlaid histograms.
        figsize : tuple
            Figure size for the plots.
        sharex : bool
            If True and the outcome is numeric, use the same x-limits across treatments.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
            (fig_distribution, fig_boxplot)
        """
        df = self.d.df
        t_col = treatment or self.d.treatment
        y_col = target or self.d.target

        if t_col not in df.columns or y_col not in df.columns:
            raise ValueError("Specified treatment/outcome columns not found in DataFrame.")

        # Determine unique treatments preserving natural sort
        treatments = pd.unique(df[t_col])

        # Distribution plot (overlayed)
        fig1 = plt.figure(figsize=figsize)
        ax1 = fig1.gca()

        # Only support numeric outcome for histogram/boxplot in this minimal implementation
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            # Fallback: draw normalized bars per treatment for categorical outcome
            # Compute frequency of outcome values per treatment and stack them side-by-side
            vals = pd.unique(df[y_col])
            vals_sorted = sorted(vals, key=lambda v: (str(type(v)), v))
            width = 0.8 / max(1, len(treatments))
            x = np.arange(len(vals_sorted))
            for i, tr in enumerate(treatments):
                sub = df[df[t_col] == tr][y_col]
                counts = pd.Series(sub).value_counts(normalize=True)
                heights = [counts.get(v, 0.0) for v in vals_sorted]
                ax1.bar(x + i * width, heights, width=width, alpha=alpha, label=str(tr))
            ax1.set_xticks(x + (len(treatments) - 1) * width / 2)
            ax1.set_xticklabels([str(v) for v in vals_sorted])
            ax1.set_ylabel("Proportion")
            ax1.set_xlabel(str(y_col))
            ax1.set_title("Target distribution by treatment (categorical)")
            ax1.legend(title=str(t_col))
        else:
            # Numeric outcome: overlay histograms/density
            # Determine common x-limits if sharex
            xmin, xmax = None, None
            if sharex:
                xmin = float(df[y_col].min())
                xmax = float(df[y_col].max())
            for tr in treatments:
                y_vals = df.loc[df[t_col] == tr, y_col].dropna().values
                if len(y_vals) == 0:
                    continue
                ax1.hist(y_vals, bins=bins, density=density, alpha=alpha, label=str(tr), range=(xmin, xmax) if sharex else None)
            ax1.set_xlabel(str(y_col))
            ax1.set_ylabel("Density" if density else "Count")
            ax1.set_title("Target distribution by treatment")
            ax1.legend(title=str(t_col))

        # Boxplot by treatment
        fig2 = plt.figure(figsize=figsize)
        ax2 = fig2.gca()
        # Create data in order of treatments
        data = [df.loc[df[t_col] == tr, y_col].dropna().values for tr in treatments]
        ax2.boxplot(data, labels=[str(tr) for tr in treatments], showfliers=True)
        ax2.set_xlabel(str(t_col))
        ax2.set_ylabel(str(y_col))
        ax2.set_title("Target by treatment (boxplot)")

        return fig1, fig2

    # ---------- one-shot driver ----------
    def design_report(self) -> Dict[str, Any]:
        """Run a full set of diagnostics and return a consolidated report.

        Returns a dict with keys:
        - health: output of data_health_check()
        - missing: output of missingness_report()
        - summaries: output of summaries()
        - treat_auc: float from treatment_predictability_auc()
        - positivity: output of positivity_check()
        - balance: output of balance_table()
        - weights: output of weight_diagnostics()
        Useful for quick inspection and logging in notebooks or pipelines.
        """
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
