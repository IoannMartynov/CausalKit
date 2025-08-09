"""
CausalForestDML implementation for estimating average treatment effects on the treated.

This module provides a function to estimate average treatment effects on the treated using EconML's CausalForestDML.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple

from econml.dml import CausalForestDML
from catboost import CatBoostRegressor, CatBoostClassifier

from causalkit.data.causaldata import CausalData


def causalforestdml(
    data: CausalData,
    model_y: Any = None,
    model_t: Any = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Estimate average treatment effects on the treated using EconML's CausalForestDML.
    
    Parameters
    ----------
    data : CausalData
        The causaldata object containing treatment, target, and cofounders variables.
    model_y : estimator, optional
        The model for fitting the outcome variable. If None, a CatBoostRegressor configured to use all CPU cores is used.
    model_t : estimator, optional
        The model for fitting the treatment variable. If None, a CatBoostRegressor configured to use all CPU cores is used.
    n_estimators : int, default 100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of the trees. If None, nodes are expanded until all leaves are pure or
        contain less than min_samples_leaf samples.
    min_samples_leaf : int, default 5
        Minimum number of samples required to be at a leaf node.
    cv : int, default 5
        Number of folds for cross-fitting.
    n_jobs : int, default -1
        Number of jobs to run in parallel. -1 means using all processors.
    random_state : int, optional
        Controls the randomness of the estimator.
    confidence_level : float, default 0.95
        The confidence level for calculating confidence intervals (between 0 and 1).
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - coefficient: The estimated average treatment effect on the treated
        - std_error: The standard error of the estimate
        - p_value: The p-value for the null hypothesis that the effect is zero
        - confidence_interval: Tuple of (lower, upper) bounds for the confidence interval
        - model: The fitted CausalForestDML object
        
    Raises
    ------
    ValueError
        If the causaldata object doesn't have treatment, target, and cofounders variables defined,
        or if the treatment variable is not binary.
        
    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import CausalData
    >>> from causalkit.inference.att import causalforestdml
    >>> 
    >>> # Generate data
    >>> df = generate_rct_data()
    >>> 
    >>> # Create causaldata object
    >>> ck = CausalData(
    ...     df=df,
    ...     outcome='outcome',
    ...     treatment='treatment',
    ...     cofounders=['age', 'invited_friend']
    ... )
    >>> 
    >>> # Estimate ATT using CausalForestDML
    >>> results = causalforestdml(ck)
    >>> print(f"ATT: {results['coefficient']:.4f}")
    >>> print(f"Standard Error: {results['std_error']:.4f}")
    >>> print(f"P-value: {results['p_value']:.4f}")
    >>> print(f"Confidence Interval: {results['confidence_interval']}")
    """
    # Validate inputs
    if data.treatment is None:
        raise ValueError("CausalData object must have a treatment variable defined")
    if data.target is None:
        raise ValueError("CausalData object must have a outcome variable defined")
    if data.cofounders is None:
        raise ValueError("CausalData object must have cofounders variables defined")
    
    # Check if treatment is binary
    unique_treatments = data.treatment.unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")
    
    # Check if treatment values are 0 and 1
    if not set(unique_treatments) == {0, 1}:
        raise ValueError("Treatment variable must have values 0 and 1")
    
    # Check confidence level
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
    
    # Set default ML models if not provided
    if model_y is None:
        model_y = CatBoostRegressor(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)
    if model_t is None:
        model_t = CatBoostClassifier(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)
    
    # Get data from CausalData object
    Y = data.target.values
    T = data.treatment.values
    X = data.cofounders.values if data.cofounders is not None else None
    
    # Create and fit CausalForestDML model
    model = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        discrete_treatment=True,  # Ensure discrete treatment for ATT estimation
    )
    
    model.fit(Y, T, X=X)
    
    # Get ATT inference results using att__inference method with T=1 (for binary treatment)
    # This returns a NormalInferenceResults object with point estimates, standard errors, and confidence intervals
    inference_results = model.att__inference(T=1)
    
    # Extract the ATT estimate, standard error, and confidence interval
    # Handle both 1D and 2D arrays
    if inference_results.point_estimate.ndim == 1:
        att = float(inference_results.point_estimate[0])  # Extract the scalar value
        std_error = float(inference_results.stderr[0])    # Extract the scalar value
    else:
        att = float(inference_results.point_estimate[0, 0])  # Extract the scalar value
        std_error = float(inference_results.stderr[0, 0])    # Extract the scalar value
    
    # Calculate p-value
    from scipy import stats
    z_value = abs(att) / std_error
    p_value = 2 * (1 - stats.norm.cdf(z_value))
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower = att - z_score * std_error
    ci_upper = att + z_score * std_error
    
    # Return results as a dictionary
    return {
        "coefficient": att,
        "std_error": std_error,
        "p_value": float(p_value),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
        "model": model
    }