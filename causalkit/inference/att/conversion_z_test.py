"""
Two-proportion z-test for conversion data in CausalData (ATT context).

Compares conversion rates between treated (T=1) and control (T=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals
(similar structure to inference.att.ttest).
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from causalkit.data.causaldata import CausalData


def conversion_z_test(data: CausalData, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    confidence_level : float, default 0.95
        The confidence level for calculating confidence intervals (between 0 and 1).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - p_value: Two-sided p-value from the z-test
        - absolute_difference: Difference in conversion rates (treated - control)
        - absolute_ci: Tuple (lower, upper) for the absolute difference CI
        - relative_difference: Percentage change relative to control rate
        - relative_ci: Tuple (lower, upper) for the relative difference CI

    Raises
    ------
    ValueError
        If treatment/outcome are missing, treatment is not binary, outcome is not binary,
        groups are empty, or confidence_level is outside (0, 1).
    """
    # Basic validation
    treatment_var = data.treatment
    target_var = data.target

    if not isinstance(treatment_var, pd.Series) or treatment_var.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(target_var, pd.Series) or target_var.empty:
        raise ValueError("causaldata object must have a outcome variable defined")

    # Treatment must be binary 0/1
    tr_unique = treatment_var.unique()
    if len(tr_unique) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")

    # Target must be binary 0/1 for conversion test
    tg_unique = set(pd.Series(target_var.unique()).dropna().tolist())
    if not tg_unique.issubset({0, 1}):
        raise ValueError("Target must be binary (0/1) for conversion_z_test")

    # Build groups using 0/1 coding on treatment
    control_mask = treatment_var == 0
    treat_mask = treatment_var == 1

    control = target_var[control_mask]
    treat = target_var[treat_mask]

    n0 = int(control.shape[0])
    n1 = int(treat.shape[0])
    if n0 < 1 or n1 < 1:
        raise ValueError("Not enough observations in one of the groups for z-test (need at least 1 per group)")

    # Counts of conversions (assumes 0/1 coding for outcome)
    x0 = float(control.sum())
    x1 = float(treat.sum())

    p0 = x0 / n0
    p1 = x1 / n1

    # Two-proportion z-test (two-sided p-value) using pooled SE under H0 for test statistic
    p_pool = (x0 + x1) / (n0 + n1)
    se_pooled = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1)))
    # Guard against zero standard error (e.g., no variance case)
    if se_pooled == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = float((p1 - p0) / se_pooled)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # Absolute difference and CI using unpooled (Wald) SE
    absolute_diff = float(p1 - p0)

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
    alpha = 1 - confidence_level
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    se_unpooled = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))
    margin = z_crit * se_unpooled
    absolute_ci = (absolute_diff - margin, absolute_diff + margin)

    # Relative difference (%) and CI via scaling by control rate
    if p0 == 0:
        relative_diff = np.inf if absolute_diff > 0 else -np.inf if absolute_diff < 0 else 0.0
        relative_ci = (np.nan, np.nan)
    else:
        relative_diff = (absolute_diff / abs(p0)) * 100.0
        relative_margin = (margin / abs(p0)) * 100.0
        relative_ci = (relative_diff - relative_margin, relative_diff + relative_margin)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
