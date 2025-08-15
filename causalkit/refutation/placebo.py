"""
Placebo / robustness checks for CausalKit.

The functions below deliberately break or perturb assumptions,
re-estimate the causal effect, and report the resulting
theta-hat and p-value so that users can judge the credibility
of their original estimate.

All helpers share the same signature:

    def refute_xxx(
        inference_fn,
        data: CausalData,
        random_state: int | None = None,
        **inference_kwargs,
    ) -> dict

Parameters
----------
inference_fn : callable
    Any CausalKit estimator (e.g. dml_att) returning a dict that
    contains keys "coefficient" and "p_value".
data : CausalData
    The original data object.
random_state : int, optional
    Seed for reproducibility.
**inference_kwargs
    Extra keyword args forwarded verbatim to `inference_fn`
    (e.g. you can tweak learners, #folds, …).

Returns
-------
dict
    {'theta': float, 'p_value': float}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any, Dict

from causalkit.data.causaldata import CausalData


def _is_binary(series: pd.Series) -> bool:
    """
    Determine if a pandas Series contains binary data (0/1 or True/False).
    
    Parameters
    ----------
    series : pd.Series
        The series to check
        
    Returns
    -------
    bool
        True if the series appears to be binary
    """
    unique_vals = set(series.dropna().unique())
    
    # Check for 0/1 binary
    if unique_vals == {0, 1} or unique_vals == {0} or unique_vals == {1}:
        return True
    
    # Check for True/False binary
    if unique_vals == {True, False} or unique_vals == {True} or unique_vals == {False}:
        return True
        
    # Check for numeric that could be treated as binary (only two distinct values)
    if len(unique_vals) == 2 and all(isinstance(x, (int, float, bool, np.integer, np.floating)) for x in unique_vals):
        return True
        
    return False


def _generate_random_outcome(original_outcome: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random outcome variables matching the distribution of the original outcome.
    
    For binary outcomes: generates random binary variables with same proportion as original
    For continuous outcomes: generates random continuous variables from normal distribution
    fitted to the original data
    
    Parameters
    ----------
    original_outcome : pd.Series
        The original outcome variable
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Generated random outcome variables
    """
    n = len(original_outcome)
    
    if _is_binary(original_outcome):
        # For binary outcome, generate random binary with same proportion
        original_rate = float(original_outcome.mean())
        return rng.binomial(1, original_rate, size=n).astype(original_outcome.dtype)
    else:
        # For continuous outcome, generate from normal distribution fitted to original data
        mean = float(original_outcome.mean())
        std = float(original_outcome.std())
        if std == 0:
            # If no variance, generate constant values equal to the mean
            return np.full(n, mean, dtype=original_outcome.dtype)
        return rng.normal(mean, std, size=n).astype(original_outcome.dtype)


def _generate_random_treatment(original_treatment: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random binary treatment variables with same proportion as original.
    
    Parameters
    ----------
    original_treatment : pd.Series
        The original treatment variable
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Generated random binary treatment variables
    """
    n = len(original_treatment)
    treatment_rate = float(original_treatment.mean())
    return rng.binomial(1, treatment_rate, size=n).astype(original_treatment.dtype)


def _run_inference(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    **kwargs,
) -> Dict[str, float]:
    """
    Helper that executes `inference_fn` and extracts the two items of interest.
    """
    res = inference_fn(data, **kwargs)
    return {"theta": float(res["coefficient"]), "p_value": float(res["p_value"])}


# ------------------------------------------------------------------
# 1. Placebo ‑- generate random outcome
# ------------------------------------------------------------------
def refute_placebo_outcome(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    random_state: int | None = None,
    **inference_kwargs,
) -> Dict[str, float]:
    """
    Generate random outcome (target) variables while keeping treatment
    and covariates intact. For binary outcomes, generates random binary
    variables with the same proportion. For continuous outcomes, generates
    random variables from a normal distribution fitted to the original data.
    A valid causal design should now yield θ ≈ 0 and a large p-value.
    """
    rng = np.random.default_rng(random_state)

    df_mod = data.get_df().copy()
    original_outcome = df_mod[data._target]
    df_mod[data._target] = _generate_random_outcome(original_outcome, rng)

    ck_mod = CausalData(
        df=df_mod,
        treatment=data._treatment,
        outcome=data._target,
        cofounders=data._cofounders,
    )
    return _run_inference(inference_fn, ck_mod, **inference_kwargs)


# ------------------------------------------------------------------
# 2. Placebo ‑- generate random treatment
# ------------------------------------------------------------------
def refute_placebo_treatment(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    random_state: int | None = None,
    **inference_kwargs,
) -> Dict[str, float]:
    """
    Generate random binary treatment variables while keeping outcome and
    covariates intact. Generates random binary treatment with the same
    proportion as the original treatment. Breaks the treatment–outcome link.
    """
    rng = np.random.default_rng(random_state)

    df_mod = data.get_df().copy()
    original_treatment = df_mod[data._treatment]
    df_mod[data._treatment] = _generate_random_treatment(original_treatment, rng)

    ck_mod = CausalData(
        df=df_mod,
        treatment=data._treatment,
        outcome=data._target,
        cofounders=data._cofounders,
    )
    return _run_inference(inference_fn, ck_mod, **inference_kwargs)


# ------------------------------------------------------------------
# 3. Subset robustness check
# ------------------------------------------------------------------
def refute_subset(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    fraction: float = 0.8,
    random_state: int | None = None,
    **inference_kwargs,
) -> Dict[str, float]:
    """
    Re-estimate the effect on a random subset (default 80 %)
    to check sample-stability of the estimate.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("`fraction` must lie in (0, 1].")

    rng = np.random.default_rng(random_state)
    df = data.get_df()
    n = len(df)
    idx = rng.choice(n, size=int(np.floor(fraction * n)), replace=False)

    df_mod = df.iloc[idx].copy()
    ck_mod = CausalData(
        df=df_mod,
        treatment=data._treatment,
        outcome=data._target,
        cofounders=data._cofounders,
    )
    return _run_inference(inference_fn, ck_mod, **inference_kwargs)