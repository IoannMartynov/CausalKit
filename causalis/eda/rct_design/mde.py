"""
Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.
"""

import numpy as np
from typing import Dict, Union, Tuple, Optional, List, Any


def calculate_mde(
    sample_size: Union[int, Tuple[int, int]],
    baseline_rate: Optional[float] = None,
    variance: Optional[Union[float, Tuple[float, float]]] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    data_type: str = 'conversion',
    ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.
    
    Parameters
    ----------
    sample_size : int or tuple of int
        Total sample size or a tuple of (control_size, treatment_size).
        If a single integer is provided, the sample will be split according to the ratio parameter.
    baseline_rate : float, optional
        Baseline conversion rate (for conversion data) or baseline mean (for continuous data).
        Required for conversion data.
    variance : float or tuple of float, optional
        Variance of the data. For conversion data, this is calculated from the baseline rate if not provided.
        For continuous data, this parameter is required.
        Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
    alpha : float, default 0.05
        Significance level (Type I error rate).
    power : float, default 0.8
        Statistical power (1 - Type II error rate).
    data_type : str, default 'conversion'
        Type of data. Either 'conversion' for binary/conversion data or 'continuous' for continuous data.
    ratio : float, default 0.5
        Ratio of the sample allocated to the control group if sample_size is a single integer.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'mde': The minimum detectable effect (absolute)
        - 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
        - 'parameters': The parameters used for the calculation
        
    Examples
    --------
    >>> # Calculate MDE for conversion data with 1000 total sample size and 10% baseline conversion rate
    >>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
    {'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
    
    >>> # Calculate MDE for continuous data with 500 samples in each group and variance of 4
    >>> calculate_mde((500, 500), variance=4, data_type='continuous')
    {'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
    
    Notes
    -----
    For conversion data, the MDE is calculated using the formula:
    MDE = (z_α/2 + z_β) * sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
    
    For continuous data, the MDE is calculated using the formula:
    MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))
    
    where:
    - z_α/2 is the critical value for significance level α
    - z_β is the critical value for power
    - p1 and p2 are the conversion rates in the control and treatment groups
    - σ1² and σ2² are the variances in the control and treatment groups
    - n1 and n2 are the sample sizes in the control and treatment groups
    """
    # Validate inputs
    if data_type not in ['conversion', 'continuous']:
        raise ValueError("data_type must be either 'conversion' or 'continuous'")
    
    if data_type == 'conversion' and baseline_rate is None:
        raise ValueError("baseline_rate is required for conversion data")
    
    if data_type == 'continuous' and variance is None:
        raise ValueError("variance is required for continuous data")
    
    # Calculate critical values
    z_alpha = abs(np.percentile(np.random.normal(0, 1, 100000), (1 - alpha / 2) * 100))
    z_beta = abs(np.percentile(np.random.normal(0, 1, 100000), power * 100))
    
    # Determine sample sizes for control and treatment groups
    if isinstance(sample_size, tuple):
        n_control, n_treatment = sample_size
    else:
        n_control = int(sample_size * ratio)
        n_treatment = sample_size - n_control
    
    # Calculate MDE based on data type
    if data_type == 'conversion':
        # For conversion data
        p_control = baseline_rate
        
        # If variance is not provided, calculate it from the baseline rate
        if variance is None:
            var_control = p_control * (1 - p_control)
            var_treatment = p_control * (1 - p_control)  # Assuming same variance for treatment
        elif isinstance(variance, tuple):
            var_control, var_treatment = variance
        else:
            var_control = var_treatment = variance
        
        # Calculate MDE
        mde = (z_alpha + z_beta) * np.sqrt((var_control / n_control) + (var_treatment / n_treatment))
        
        # Calculate relative MDE
        mde_relative = mde / baseline_rate if baseline_rate > 0 else None
        
    else:  # data_type == 'continuous'
        # For continuous data
        if isinstance(variance, tuple):
            var_control, var_treatment = variance
        else:
            var_control = var_treatment = variance
        
        # Calculate MDE
        mde = (z_alpha + z_beta) * np.sqrt((var_control / n_control) + (var_treatment / n_treatment))
        
        # Calculate relative MDE if baseline_rate is provided
        mde_relative = mde / baseline_rate if baseline_rate is not None and baseline_rate != 0 else None
    
    # Prepare the result dictionary
    result = {
        'mde': mde,
        'mde_relative': mde_relative,
        'parameters': {
            'sample_size': {
                'total': n_control + n_treatment,
                'control': n_control,
                'treatment': n_treatment
            },
            'alpha': alpha,
            'power': power,
            'data_type': data_type
        }
    }
    
    # Add data-type specific parameters
    if data_type == 'conversion':
        result['parameters']['baseline_rate'] = baseline_rate
    else:  # data_type == 'continuous'
        result['parameters']['variance'] = {
            'control': var_control,
            'treatment': var_treatment
        }
        if baseline_rate is not None:
            result['parameters']['baseline_mean'] = baseline_rate
    
    return result