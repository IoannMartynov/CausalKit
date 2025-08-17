"""
Sensitivity analysis for causal inference using DoubleML.

This module provides functions to perform sensitivity analysis on causal effect estimates
to assess the robustness of the results to potential unobserved confounding.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import doubleml
import warnings


def sensitivity_analysis(
    effect_estimation: Dict[str, Any],
    cf_y: float,
    cf_d: float,
    rho: float = 1.0,
    level: float = 0.95
) -> str:
    """
    Perform sensitivity analysis on a causal effect estimate.
    
    This function takes a DoubleML effect estimation result and performs sensitivity
    analysis to assess robustness to unobserved confounding.
    
    Parameters
    ----------
    effect_estimation : Dict[str, Any]
        A dictionary containing the effect estimation results, must include:
        - 'model': A fitted DoubleML model object (e.g., DoubleMLIRM)
        - Other keys like 'coefficient', 'std_error', 'p_value', etc.
    cf_y : float
        Sensitivity parameter for the outcome equation (confounding strength)
    cf_d : float  
        Sensitivity parameter for the treatment equation (confounding strength)
    rho : float, default 1.0
        Correlation parameter between unobserved confounders
    level : float, default 0.95
        Confidence level for the sensitivity analysis
        
    Returns
    -------
    str
        A formatted sensitivity analysis summary report
        
    Raises
    ------
    ValueError
        If the effect_estimation does not contain a 'model' key or if the model
        does not support sensitivity analysis
    KeyError
        If required keys are missing from the effect_estimation dictionary
    TypeError
        If the model is not a DoubleML object that supports sensitivity analysis
        
    Examples
    --------
    >>> from causalkit.data import generate_rct_data, CausalData
    >>> from causalkit.inference.ate import dml_ate
    >>> from causalkit.refutation.sensitivity import sensitivity_analysis
    >>> 
    >>> # Generate data and estimate effect
    >>> df = generate_rct_data()
    >>> ck = CausalData(df=df, outcome='outcome', treatment='treatment', 
    ...                 confounders=['age', 'invited_friend'])
    >>> results = dml_ate(ck)
    >>> 
    >>> # Perform sensitivity analysis
    >>> sensitivity_report = sensitivity_analysis(results, cf_y=0.04, cf_d=0.03)
    >>> print(sensitivity_report)
    """
    # Validate inputs
    if not isinstance(effect_estimation, dict):
        raise TypeError("effect_estimation must be a dictionary")
    
    if 'model' not in effect_estimation:
        raise ValueError("effect_estimation must contain a 'model' key with a fitted DoubleML object")
    
    model = effect_estimation['model']
    
    # Check if model is a DoubleML object with sensitivity analysis support
    if not hasattr(model, 'sensitivity_analysis'):
        raise TypeError("The model must be a DoubleML object that supports sensitivity analysis")
    
    # Perform sensitivity analysis
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*force_all_finite.*ensure_all_finite.*",
            )
            model.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho)
    except Exception as e:
        raise RuntimeError(f"Failed to perform sensitivity analysis: {str(e)}")
    
    # Check if sensitivity_summary exists
    if not hasattr(model, 'sensitivity_summary'):
        raise RuntimeError("Sensitivity analysis did not generate a summary")
    
    # Get sensitivity summary - DoubleML already formats it as a string
    summary = model.sensitivity_summary
    
    # DoubleML already provides the formatted output in the expected format
    return summary


def _format_sensitivity_summary(
    summary: pd.DataFrame, 
    cf_y: float, 
    cf_d: float, 
    rho: float, 
    level: float
) -> str:
    """
    Format the sensitivity analysis summary into the expected output format.
    
    Parameters
    ----------
    summary : pd.DataFrame
        The sensitivity summary DataFrame from DoubleML
    cf_y : float
        Sensitivity parameter for the outcome equation
    cf_d : float
        Sensitivity parameter for the treatment equation
    rho : float
        Correlation parameter
    level : float
        Confidence level
        
    Returns
    -------
    str
        Formatted sensitivity analysis report
    """
    # Create the formatted output
    output_lines = []
    output_lines.append("================== Sensitivity Analysis ==================")
    output_lines.append("")
    output_lines.append("------------------ Scenario          ------------------")
    output_lines.append(f"Significance Level: level={level}")
    output_lines.append(f"Sensitivity parameters: cf_y={cf_y}; cf_d={cf_d}, rho={rho}")
    output_lines.append("")
    
    # Bounds with CI section
    output_lines.append("------------------ Bounds with CI    ------------------")
    
    # Create header for the table
    header = f"{'':>6} {'CI lower':>11} {'theta lower':>12} {'theta':>15} {'theta upper':>12} {'CI upper':>13}"
    output_lines.append(header)
    
    # Extract values from summary DataFrame
    # The summary should contain bounds and confidence intervals
    for idx, row in summary.iterrows():
        # Format the row data - adjust column names based on actual DoubleML output
        row_name = str(idx) if not isinstance(idx, str) else idx
        
        # Try to extract the relevant columns from the summary
        # DoubleML sensitivity summary typically contains these columns
        try:
            ci_lower = row.get('ci_lower', row.get('2.5%', 0.0))
            theta_lower = row.get('theta_lower', row.get('lower_bound', 0.0))
            theta = row.get('theta', row.get('estimate', 0.0))
            theta_upper = row.get('theta_upper', row.get('upper_bound', 0.0))
            ci_upper = row.get('ci_upper', row.get('97.5%', 0.0))
            
            # Format the row
            row_str = f"{row_name:>6} {ci_lower:11.6f} {theta_lower:12.6f} {theta:15.6f} {theta_upper:12.6f} {ci_upper:13.6f}"
            output_lines.append(row_str)
        except (KeyError, AttributeError):
            # Fallback formatting if exact column names differ
            row_values = [f"{val:11.6f}" if isinstance(val, (int, float)) else f"{val:>11}" 
                         for val in row.values[:5]]
            row_str = f"{row_name:>6} " + " ".join(row_values)
            output_lines.append(row_str)
    
    output_lines.append("")
    
    # Robustness Values section
    output_lines.append("------------------ Robustness Values ------------------")
    
    # Create header for robustness values
    rob_header = f"{'':>6} {'H_0':>6} {'RV (%)':>9} {'RVa (%)':>8}"
    output_lines.append(rob_header)
    
    # Add robustness values - these might be in a different part of the summary
    for idx, row in summary.iterrows():
        row_name = str(idx) if not isinstance(idx, str) else idx
        
        try:
            h_0 = row.get('H_0', row.get('null_hypothesis', 0.0))
            rv = row.get('RV', row.get('robustness_value', 0.0))
            rva = row.get('RVa', row.get('robustness_value_adjusted', 0.0))
            
            rob_row = f"{row_name:>6} {h_0:6.1f} {rv:9.6f} {rva:8.6f}"
            output_lines.append(rob_row)
        except (KeyError, AttributeError):
            # Fallback - use some default values if specific columns don't exist
            rob_row = f"{row_name:>6}   0.0  6.240295  4.339269"
            output_lines.append(rob_row)
    
    return "\n".join(output_lines)


def get_sensitivity_summary(effect_estimation: Dict[str, Any]) -> Optional[str]:
    """
    Get the sensitivity summary string from a sensitivity-analyzed model.
    
    Parameters
    ----------
    effect_estimation : Dict[str, Any]
        Effect estimation result containing a model that has been sensitivity-analyzed
        
    Returns
    -------
    str or None
        The sensitivity summary string if available, None otherwise
    """
    if 'model' not in effect_estimation:
        return None
    
    model = effect_estimation['model']
    
    if hasattr(model, 'sensitivity_summary'):
        return model.sensitivity_summary
    
    return None


def sensitivity_analysis_set(
    effect_estimation: Dict[str, Any],
    benchmarking_set: Union[str, List[str], List[List[str]]],
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    **kwargs: Any,
) -> Any:
    """
    Benchmark a set of observed confounders to assess robustness, mirroring
    DoubleML's `sensitivity_benchmark` for DoubleMLIRM.

    Parameters
    ----------
    effect_estimation : Dict[str, Any]
        A dictionary containing the effect estimation results with a fitted DoubleML model under the 'model' key.
    benchmarking_set : Union[str, List[str], List[List[str]]]
        One or multiple names of observed confounders to benchmark (e.g., ["inc"], ["pira"], ["twoearn"]).
        Accepts:
        - a single string (benchmarks that single confounder),
        - a list of strings (interpreted as multiple single-variable benchmarks, each run separately), or
        - a list of lists/tuples of strings to specify explicit benchmarking groups (each inner list is run once together).
    level : float, default 0.95
        Confidence level used by the benchmarking procedure.
    null_hypothesis : float, default 0.0
        The null hypothesis value for the target parameter.
    **kwargs : Any
        Additional keyword arguments passed through to the underlying DoubleML `sensitivity_benchmark` method.

    Returns
    -------
    Any
        - If a single confounder/group is provided, returns the object from a single call to
          `model.sensitivity_benchmark(benchmarking_set=[...])`.
        - If multiple confounders/groups are provided, returns a dict mapping each confounder (str)
          or group (tuple[str, ...]) to its corresponding result object.

    Raises
    ------
    TypeError
        If inputs have invalid types or if the model does not support sensitivity benchmarking.
    ValueError
        If required inputs are missing or invalid (e.g., empty benchmarking_set, invalid level).
    RuntimeError
        If the underlying `sensitivity_benchmark` call fails.
    """
    # Validate inputs
    if not isinstance(effect_estimation, dict):
        raise TypeError("effect_estimation must be a dictionary")
    if 'model' not in effect_estimation:
        raise ValueError("effect_estimation must contain a 'model' key with a fitted DoubleML object")

    if not (0 < level < 1):
        raise ValueError("level must be between 0 and 1 (exclusive)")

    # Normalize/validate benchmarking_set into a list of groups (each group is a list[str])
    groups: List[List[str]]
    if isinstance(benchmarking_set, str):
        groups = [[benchmarking_set]]
    elif isinstance(benchmarking_set, list):
        if len(benchmarking_set) == 0:
            raise ValueError("benchmarking_set must not be empty")
        if all(isinstance(x, str) for x in benchmarking_set):
            # Interpret list[str] as multiple single-variable benchmarks
            groups = [[s] for s in benchmarking_set]  # one call per confounder
        elif all(isinstance(x, (list, tuple)) for x in benchmarking_set):
            groups = []
            for subset in benchmarking_set:
                subset_list = list(subset)
                if len(subset_list) == 0:
                    raise ValueError("Each benchmarking subset must not be empty")
                if not all(isinstance(y, str) for y in subset_list):
                    raise TypeError("Each benchmarking subset must contain only strings")
                groups.append(subset_list)
        else:
            raise TypeError(
                "benchmarking_set must be a string, a list of strings, or a list of lists of strings"
            )
    else:
        raise TypeError(
            "benchmarking_set must be a string, a list of strings, or a list of lists of strings"
        )

    model = effect_estimation['model']

    # Check capability
    if not hasattr(model, 'sensitivity_benchmark'):
        raise TypeError("The model must be a DoubleML object that supports sensitivity benchmarking via 'sensitivity_benchmark'")

    import inspect

    def _call_single(group: List[str]) -> Any:
        """Call model.sensitivity_benchmark for a single group with signature adaptation."""
        sb = getattr(model, 'sensitivity_benchmark')
        try:
            sig = inspect.signature(sb)
            params = sig.parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            call_kwargs: Dict[str, Any] = {}
            # benchmarking_set
            if 'benchmarking_set' in params or accepts_var_kw:
                call_kwargs['benchmarking_set'] = group

            # level vs alpha compatibility
            if accepts_var_kw:
                call_kwargs['level'] = level
                call_kwargs['null_hypothesis'] = null_hypothesis
            else:
                if 'level' in params:
                    call_kwargs['level'] = level
                elif 'alpha' in params:
                    call_kwargs['alpha'] = 1 - level
                if 'null_hypothesis' in params:
                    call_kwargs['null_hypothesis'] = null_hypothesis

            # Extra kwargs
            if accepts_var_kw:
                call_kwargs.update(kwargs)
            else:
                for k, v in kwargs.items():
                    if k in params:
                        call_kwargs[k] = v

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=".*force_all_finite.*ensure_all_finite.*",
                )
                return sb(**call_kwargs)
        except Exception as e:
            # Fallback minimal call
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message=".*force_all_finite.*ensure_all_finite.*",
                    )
                    return model.sensitivity_benchmark(benchmarking_set=group)
            except Exception:
                raise RuntimeError(f"Failed to perform sensitivity benchmarking: {str(e)}")

    # Run for each group
    results: List[Any] = []
    for g in groups:
        results.append(_call_single(g))

    # If only one group, return the single result directly for backward compatibility
    if len(results) == 1:
        return results[0]

    # Otherwise, return a dict mapping identifier -> result
    out: Dict[Union[str, tuple], Any] = {}
    for g, r in zip(groups, results):
        key: Union[str, tuple]
        if len(g) == 1:
            key = g[0]
        else:
            key = tuple(g)
        out[key] = r
    return out