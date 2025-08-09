"""
Group Average Treatment Effect (GATE) estimation using DoubleML orthogonal signals.

This module provides a function that, given a (possibly filtered) CausalData object,
fits a DoubleML IRM model, computes per-observation CATEs (orthogonal signals),
forms groups (by default CATE quintiles), and returns group-level estimates
(theta), standard errors, p-values, and confidence intervals.

It prefers DoubleML's native `gate()` and `confint()` methods if available;
otherwise falls back to a simple normal approximation using the group mean of
orthogonal signals and its standard error.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

import doubleml as dml
from catboost import CatBoostRegressor, CatBoostClassifier

from causalkit.data.causaldata import CausalData


def _fit_doubleml_irm(
    data: CausalData,
    ml_g: Optional[Any],
    ml_m: Optional[Any],
    n_folds: int,
    n_rep: int,
):
    if ml_g is None:
        ml_g = CatBoostRegressor(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)
    if ml_m is None:
        ml_m = CatBoostClassifier(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)

    df = data.get_df()
    dml_data = dml.DoubleMLData(
        df,
        y_col=data._target,
        d_cols=data._treatment,
        x_cols=data._cofounders,
    )

    obj = dml.DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score="ATE",
    )
    # Suppress scikit-learn FutureWarning about 'force_all_finite' rename during fit
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
            category=FutureWarning,
        )
        obj.fit()
    return obj, df


def _extract_signals(obj: Any) -> np.ndarray:
    # Prefer private attribute if present
    if hasattr(obj, "_orthogonal_signals"):
        sig = np.asarray(obj._orthogonal_signals).reshape(-1)
        return sig
    # Fallback via psi_elements if available
    if hasattr(obj, "psi_elements") and isinstance(obj.psi_elements, dict) and "psi_b" in obj.psi_elements:
        arr = np.asarray(obj.psi_elements["psi_b"])  # shape could include folds/reps
        if arr.ndim == 1:
            return arr
        axes = tuple(range(1, arr.ndim))
        return np.nanmean(arr, axis=axes)
    raise AttributeError("Could not extract orthogonal signals from DoubleMLIRM object.")


def gate_esimand(
    data: CausalData,
    groups: Optional[Union[pd.Series, pd.DataFrame]] = None,
    n_groups: int = 5,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Estimate Group Average Treatment Effects (GATEs) by grouping observations
    using CATE-based quantiles unless custom groups are provided.

    Parameters
    ----------
    data : CausalData
        The (possibly filtered) CausalData object. Filtering should be done by
        subsetting data.df before constructing CausalData, or by preparing a
        filtered CausalData instance.
    groups : pd.Series or pd.DataFrame, optional
        Group assignments per observation. If a Series is passed, it will be
        used as a single column named 'q'. If a DataFrame, it should contain a
        single column specifying groups. If None, groups are formed by
        pd.qcut over the in-sample CATEs into `n_groups` quantiles labeled 0..n_groups-1.
    n_groups : int, default 5
        Number of quantile groups if `groups` is None.
    ml_g, ml_m, n_folds, n_rep :
        Learners and DoubleML cross-fitting controls (as in ATE/ATT).
    confidence_level : float, default 0.95
        Confidence level for two-sided normal-approximation intervals.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
          - group: group label
          - n: group size
          - theta: estimated group average treatment effect
          - std_error: standard error (normal approx if fallback path)
          - p_value: two-sided p-value for H0: theta=0
          - ci_lower, ci_upper: confidence interval bounds
    """
    # Validate inputs similar to other inference functions
    if data.treatment is None:
        raise ValueError("CausalData object must have a treatment variable defined")
    if data.target is None:
        raise ValueError("CausalData object must have a outcome variable defined")
    if data.cofounders is None:
        raise ValueError("CausalData object must have cofounders variables defined")

    # Binary treatment check {0,1}
    unique_treatments = pd.Series(data.treatment).unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")
    if set(unique_treatments) != {0, 1}:
        raise ValueError("Treatment variable must have values 0 and 1")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")

    # Fit DoubleML model and extract signals
    obj, df = _fit_doubleml_irm(data, ml_g, ml_m, n_folds, n_rep)
    cate_hat = _extract_signals(obj)

    # Prepare groups
    if groups is None:
        # Build quantile groups from CATEs
        try:
            q = pd.qcut(cate_hat, n_groups, labels=False, duplicates="drop")
        except ValueError:
            # In case of too many ties, fall back to cut on unique bins
            q = pd.Series(pd.cut(cate_hat, n_groups, labels=False, duplicates="drop"))
        groups_df = pd.DataFrame({"q": q})
    else:
        if isinstance(groups, pd.Series):
            groups_df = groups.to_frame(name="q")
        elif isinstance(groups, pd.DataFrame):
            if groups.shape[1] != 1:
                raise ValueError("groups DataFrame must have exactly one column")
            groups_df = groups.copy()
            groups_df.columns = ["q"]
        else:
            raise TypeError("groups must be a pandas Series or DataFrame if provided")
        if len(groups_df) != len(df):
            raise ValueError("groups length must match number of observations in data")

    # Try DoubleML native gate() if available
    if hasattr(obj, "gate"):
        try:
            gate_obj = obj.gate(groups=groups_df)
            # confint for simultaneous intervals (if available)
            ci_df = None
            try:
                ci_df = gate_obj.confint(level=confidence_level)
            except Exception:
                ci_df = None

            # Extract estimates
            # gate_obj might expose .coef, .se, .pval similar to DoubleML objects
            if hasattr(gate_obj, "coef") and hasattr(gate_obj, "se") and hasattr(gate_obj, "pval"):
                theta = np.asarray(gate_obj.coef).reshape(-1)
                se = np.asarray(gate_obj.se).reshape(-1)
                pval = np.asarray(gate_obj.pval).reshape(-1)
                # Map groups to unique sorted labels
                labels, counts = np.unique(groups_df["q"].to_numpy(), return_counts=True)
                z = norm.ppf(1 - (1 - confidence_level) / 2)
                ci_lower = theta - z * se
                ci_upper = theta + z * se
                if isinstance(ci_df, pd.DataFrame) and ci_df.shape[0] == theta.shape[0]:
                    # prefer provided CIs if shapes align
                    ci_lower = ci_df.iloc[:, 0].to_numpy()
                    ci_upper = ci_df.iloc[:, 1].to_numpy()
                out = pd.DataFrame(
                    {
                        "group": labels,
                        "n": counts,
                        "theta": theta,
                        "std_error": se,
                        "p_value": pval,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )
                return out.sort_values("group").reset_index(drop=True)
        except Exception:
            # Fall back to manual computation below
            pass

    # Manual group-wise estimation from orthogonal signals
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    groups_df = groups_df.reset_index(drop=True)
    sig_ser = pd.Series(cate_hat).reset_index(drop=True)
    df_g = pd.concat([groups_df, sig_ser.rename("signal")], axis=1)
    agg = df_g.groupby("q")["signal"].agg(["mean", "count", "std"]).reset_index()
    # Handle std = NaN (groups of size 1) by setting SE to inf (pval=1, CI wide)
    se = (agg["std"] / np.sqrt(agg["count"].clip(lower=1))).to_numpy()
    theta = agg["mean"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.where(se > 0, theta / se, 0.0)
        pval = 2 * (1 - norm.cdf(np.abs(zstat)))
        ci_lower = theta - z * se
        ci_upper = theta + z * se
    out = pd.DataFrame(
        {
            "group": agg["q"].to_numpy(),
            "n": agg["count"].to_numpy(),
            "theta": theta,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )
    return out.sort_values("group").reset_index(drop=True)
