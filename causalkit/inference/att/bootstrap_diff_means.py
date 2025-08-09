"""
Bootstrap difference-in-means inference for CausalData (ATT context).

Computes the ATT-style difference in means (treated - control) and provides:
- Two-sided p-value using a normal approximation with bootstrap standard error
- Percentile confidence interval for the absolute difference
- Relative difference (%) and corresponding CI relative to control mean

Input:
- data: CausalData
- confidence_level: float in (0, 1), default 0.95
- n_simul: number of bootstrap simulations (int > 0), default 10000

Output: dict with the same keys as ttest:
- p_value
- absolute_difference
- absolute_ci: (low, high)
- relative_difference
- relative_ci: (low, high)
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from causalkit.data.causaldata import CausalData


def bootstrap_diff_means(
    data: CausalData,
    confidence_level: float = 0.95,
    n_simul: int = 10000,
) -> Dict[str, Any]:
    """
    Bootstrap inference for difference in means between treated (T=1) and control (T=0).

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and target variables.
    confidence_level : float, default 0.95
        Confidence level for the percentile confidence interval (0 < level < 1).
    n_simul : int, default 10000
        Number of bootstrap resamples.

    Returns
    -------
    Dict[str, Any]
        Dictionary with p_value, absolute_difference, absolute_ci, relative_difference, relative_ci
        (matching the structure of inference.att.ttest).

    Raises
    ------
    ValueError
        If inputs are invalid, treatment is not binary, or groups are empty.
    """
    # Validate inputs
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
    if not isinstance(n_simul, int) or n_simul <= 0:
        raise ValueError("n_simul must be a positive integer")

    treatment = data.treatment
    target = data.target

    if not isinstance(treatment, pd.Series) or treatment.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(target, pd.Series) or target.empty:
        raise ValueError("causaldata object must have a target variable defined")

    uniq = treatment.unique()
    if len(uniq) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")

    control = target[treatment == 0]
    treated = target[treatment == 1]

    n0 = int(control.shape[0])
    n1 = int(treated.shape[0])
    if n0 < 1 or n1 < 1:
        raise ValueError("Not enough observations in one of the groups for bootstrap (need at least 1 per group)")

    control_mean = float(control.mean())
    treated_mean = float(treated.mean())
    abs_diff = float(treated_mean - control_mean)

    # Prepare for bootstrap: indices for resampling within each group
    ctrl_vals = control.to_numpy()
    trt_vals = treated.to_numpy()
    rng = np.random.default_rng()

    # Vectorized bootstrap using random integers for indices
    ctrl_idx = rng.integers(0, n0, size=(n_simul, n0))
    trt_idx = rng.integers(0, n1, size=(n_simul, n1))

    ctrl_boot_means = ctrl_vals[ctrl_idx].mean(axis=1)
    trt_boot_means = trt_vals[trt_idx].mean(axis=1)
    boot_diffs = trt_boot_means - ctrl_boot_means

    # Percentile CI for absolute difference
    alpha = 1 - confidence_level
    lower = float(np.quantile(boot_diffs, alpha / 2))
    upper = float(np.quantile(boot_diffs, 1 - alpha / 2))
    absolute_ci = (lower, upper)

    # p-value using bootstrap SE and normal approximation
    se_boot = float(np.std(boot_diffs, ddof=1))
    if se_boot == 0:
        p_value = 1.0
    else:
        z = abs_diff / se_boot
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Relative effects and CI by scaling
    if control_mean == 0:
        relative_diff = np.inf if abs_diff > 0 else 0.0 if abs_diff == 0 else -np.inf
        relative_ci = (np.nan, np.nan)
    else:
        relative_diff = (abs_diff / abs(control_mean)) * 100.0
        rel_lower = (lower / abs(control_mean)) * 100.0
        rel_upper = (upper / abs(control_mean)) * 100.0
        relative_ci = (float(rel_lower), float(rel_upper))

    return {
        "p_value": float(p_value),
        "absolute_difference": float(abs_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
