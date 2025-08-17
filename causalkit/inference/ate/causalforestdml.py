"""
CausalForestDML implementation for estimating average treatment effects.

This module provides a function to estimate average treatment effects using EconML's CausalForestDML.
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
    Estimate average treatment effects using EconML's CausalForestDML.
    
    Parameters
    ----------
    data : CausalData
        The causaldata object containing treatment, target, and confounders variables.
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
        - coefficient: The estimated average treatment effect
        - std_error: The standard error of the estimate
        - p_value: The p-value for the null hypothesis that the effect is zero
        - confidence_interval: Tuple of (lower, upper) bounds for the confidence interval
        - model: The fitted CausalForestDML object
        
    Raises
    ------
    ValueError
        If the causaldata object doesn't have treatment, target, and confounders variables defined,
        or if the treatment variable is not binary.
        
    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import CausalData
    >>> from causalkit.inference.ate import causalforestdml
    >>> 
    >>> # Generate data
    >>> df = generate_rct_data()
    >>> 
    >>> # Create causaldata object
    >>> ck = CausalData(
    ...     df=df,
    ...     outcome='outcome',
    ...     treatment='treatment',
    ...     confounders=['age', 'invited_friend']
    ... )
    >>> 
    >>> # Estimate ATE using CausalForestDML
    >>> results = causalforestdml(ck)
    >>> print(f"ATE: {results['coefficient']:.4f}")
    >>> print(f"Standard Error: {results['std_error']:.4f}")
    >>> print(f"P-value: {results['p_value']:.4f}")
    >>> print(f"Confidence Interval: {results['confidence_interval']}")
    """
    # Validate inputs
    if data.treatment is None:
        raise ValueError("CausalData object must have a treatment variable defined")
    if data.target is None:
        raise ValueError("CausalData object must have a outcome variable defined")
    if data.confounders is None:
        raise ValueError("CausalData object must have confounders variables defined")
    
    # # Check if treatment is binary
    # unique_treatments = data.treatment.unique()
    # if len(unique_treatments) != 2:
    #     raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")
    #
    # # Check if treatment values are 0 and 1
    # if not set(unique_treatments) == {0, 1}:
    #     raise ValueError("Treatment variable must have values 0 and 1")
    
    # Check confidence level
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
    
    # Set default ML models if not provided
    if model_y is None:
        model_y = CatBoostRegressor(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False, allow_writing_files=False)
    if model_t is None:
        # For binary treatments, a classifier is more appropriate
        model_t = CatBoostClassifier(iterations=100, depth=5, thread_count=-1, verbose=False, allow_writing_files=False)
    
    # Get data from CausalData object
    Y = data.target.values
    T = data.treatment.values
    conf_list = data.confounders
    if conf_list:
        X = data.get_df(include_treatment=False, include_target=False, include_confounders=True).values
    else:
        X = None
    
    # Create and fit CausalForestDML model
    model = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        cv=cv,
        discrete_treatment=True,
        random_state=random_state,
    )
    
    model.fit(Y, T, X=X)
    
    # Compute ATE and its confidence interval using EconML's built-in methods
    alpha = 1 - confidence_level
    ate = float(model.ate(X=X))
    ci_lower, ci_upper = model.ate_interval(X=X, alpha=alpha)
    ci_lower = float(ci_lower)
    ci_upper = float(ci_upper)
    
    # Derive standard error from CI width assuming normal approximation
    from scipy import stats
    z_score = stats.norm.ppf(1 - alpha/2)
    std_error = (ci_upper - ci_lower) / (2 * z_score) if z_score > 0 else 0.0
    
    # Compute two-sided p-value for H0: ate = 0
    z_value = abs(ate) / std_error if std_error > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(z_value)) if np.isfinite(z_value) else 0.0
    
    return {
        "coefficient": ate,
        "std_error": float(std_error),
        "p_value": float(p_value),
        "confidence_interval": (ci_lower, ci_upper),
        "model": model
    }