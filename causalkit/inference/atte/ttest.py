"""
T-test inference for causaldata objects (ATT context).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

from causalkit.data.causaldata import CausalData


def ttest(data: CausalData, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Perform a t-test on a CausalData object to compare the outcome variable between
    treated (T=1) and control (T=0) groups. Returns differences and confidence intervals.
    
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
        - p_value: The p-value from the t-test
        - absolute_difference: The absolute difference between treatment and control means
        - absolute_ci: Tuple of (lower, upper) bounds for the absolute difference confidence interval
        - relative_difference: The relative difference (percentage change) between treatment and control means
        - relative_ci: Tuple of (lower, upper) bounds for the relative difference confidence interval
        
    Raises
    ------
    ValueError
        If the CausalData object doesn't have both treatment and outcome variables defined,
        or if the treatment variable is not binary.
    """
    # Basic validation: ensure treatment and outcome are proper Series and non-empty
    treatment_var = data.treatment
    target_var = data.target

    if not isinstance(treatment_var, pd.Series) or treatment_var.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(target_var, pd.Series) or target_var.empty:
        raise ValueError("causaldata object must have a outcome variable defined")

    # Ensure binary treatment
    unique_treatments = treatment_var.unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")

    # Build groups by conventional 0/1 coding
    control_data = target_var[treatment_var == 0]
    treatment_data = target_var[treatment_var == 1]

    # Independent two-sample t-test (pooled variance by default equal_var=True)
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=True)

    # Means
    control_mean = float(control_data.mean())
    treatment_mean = float(treatment_data.mean())

    # Absolute difference (ATT style: treated - control)
    absolute_diff = treatment_mean - control_mean

    # Standard error using pooled variance
    n1 = int(len(treatment_data))
    n2 = int(len(control_data))
    s1_squared = float(treatment_data.var(ddof=1))
    s2_squared = float(control_data.var(ddof=1))

    # Guard against degenerate cases with very small groups
    if n1 < 2 or n2 < 2:
        raise ValueError("Not enough observations in one of the groups for t-test (need at least 2 per group)")

    pooled_var = ((n1 - 1) * s1_squared + (n2 - 1) * s2_squared) / (n1 + n2 - 2)
    se_diff = float(np.sqrt(pooled_var * (1 / n1 + 1 / n2)))

    # Confidence interval
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
    alpha = 1 - confidence_level
    df = n1 + n2 - 2
    t_critical = float(stats.t.ppf(1 - alpha / 2, df))

    margin_of_error = t_critical * se_diff
    absolute_ci = (absolute_diff - margin_of_error, absolute_diff + margin_of_error)

    # Relative difference (%), relative CI via delta method on denominator
    if control_mean == 0:
        relative_diff = np.inf if absolute_diff > 0 else -np.inf if absolute_diff < 0 else 0.0
        relative_ci = (np.nan, np.nan)
    else:
        relative_diff = (absolute_diff / abs(control_mean)) * 100.0
        relative_margin = (margin_of_error / abs(control_mean)) * 100.0
        relative_ci = (relative_diff - relative_margin, relative_diff + relative_margin)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
