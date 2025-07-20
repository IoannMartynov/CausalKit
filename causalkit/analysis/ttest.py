"""
T-test analysis for causaldata objects.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Union, Tuple

from causalkit.data.causaldata import causaldata


def ttest(data: causaldata, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Perform a t-test on a causaldata object to compare the target variable between treatment groups.
    
    Parameters
    ----------
    data : causaldata
        The causaldata object containing treatment and target variables.
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
        If the causaldata object doesn't have both treatment and target variables defined,
        or if the treatment variable is not binary.
        
    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import causaldata
    >>> from causalkit.analysis import ttest
    >>> 
    >>> # Generate data
    >>> df = generate_rct_data()
    >>> 
    >>> # Create causaldata object
    >>> ck = causaldata(
    ...     df=df,
    ...     target='target',
    ...     treatment='treatment'
    ... )
    >>> 
    >>> # Perform t-test
    >>> results = ttest(ck)
    >>> print(f"P-value: {results['p_value']:.4f}")
    >>> print(f"Absolute difference: {results['absolute_difference']:.4f}")
    >>> print(f"Absolute CI: {results['absolute_ci']}")
    >>> print(f"Relative difference: {results['relative_difference']:.2f}%")
    >>> print(f"Relative CI: {results['relative_ci']}")
    """
    # Validate inputs
    if data.treatment is None:
        raise ValueError("causaldata object must have a treatment variable defined")
    if data.target is None:
        raise ValueError("causaldata object must have a target variable defined")
    
    # Extract treatment and target data
    treatment_var = data.treatment
    target_var = data.target
    
    # Check if treatment is binary
    unique_treatments = treatment_var.unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")
    
    # Identify control and treatment groups
    control_value = unique_treatments[0]
    treatment_value = unique_treatments[1]
    
    # Split data into control and treatment groups
    control_data = target_var[treatment_var == 0]
    treatment_data = target_var[treatment_var == 1]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=True)
    
    # Calculate means
    control_mean = control_data.mean()
    treatment_mean = treatment_data.mean()
    
    # Calculate absolute difference
    absolute_diff = treatment_mean - control_mean
    
    # Calculate standard error of the difference
    n1 = len(treatment_data)
    n2 = len(control_data)
    s1_squared = treatment_data.var(ddof=1)
    s2_squared = control_data.var(ddof=1)
    
    # Pooled variance
    pooled_var = ((n1 - 1) * s1_squared + (n2 - 1) * s2_squared) / (n1 + n2 - 2)
    
    # Standard error of the difference
    se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # Calculate t-critical value for the given confidence level
    alpha = 1 - confidence_level
    df = n1 + n2 - 2  # Degrees of freedom
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Calculate confidence interval for absolute difference
    margin_of_error = t_critical * se_diff
    absolute_ci = (absolute_diff - margin_of_error, absolute_diff + margin_of_error)
    
    # Calculate relative difference (percentage change)
    if control_mean == 0:
        # Handle division by zero
        relative_diff = np.inf if absolute_diff > 0 else -np.inf if absolute_diff < 0 else 0
        relative_ci = (np.nan, np.nan)  # Can't calculate CI when denominator is zero
    else:
        relative_diff = (absolute_diff / abs(control_mean)) * 100
        
        # Calculate confidence interval for relative difference
        relative_margin = (margin_of_error / abs(control_mean)) * 100
        relative_ci = (relative_diff - relative_margin, relative_diff + relative_margin)

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")

    # Return results as a dictionary
    return {
        "p_value": p_value,
        "absolute_difference": absolute_diff,
        "absolute_ci": absolute_ci,
        "relative_difference": relative_diff,
        "relative_ci": relative_ci
    }

