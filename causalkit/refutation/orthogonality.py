"""
AIPW orthogonality diagnostics for DoubleML estimators.

This module implements comprehensive orthogonality diagnostics for AIPW/IRM-based
estimators like dml_ate and dml_att to validate the key assumptions required 
for valid causal inference. Based on the efficient influence function (EIF) framework.

Key diagnostics implemented:
- Out-of-sample moment check (non-tautological)
- Orthogonality (Gateaux derivative) tests
- Influence diagnostics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Any, Dict, List, Tuple, Optional
from scipy import stats
from sklearn.model_selection import KFold

from causalkit.data.causaldata import CausalData


def aipw_score(y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, g: np.ndarray, theta_hat: float) -> np.ndarray:
    """
    Compute AIPW (Augmented Inverse Propensity Weighting) scores.
    
    The AIPW score is the efficient influence function (EIF) for the ATE:
    ψ_i(θ,η) = [m1(X_i) - m0(X_i)] + D_i(Y_i - m1(X_i))/g(X_i) - (1-D_i)(Y_i - m0(X_i))/(1-g(X_i)) - θ
    
    Parameters
    ----------
    y : np.ndarray
        Observed outcomes
    d : np.ndarray  
        Binary treatment indicator (0/1)
    m0 : np.ndarray
        Predicted outcomes under control E[Y|X,D=0]
    m1 : np.ndarray
        Predicted outcomes under treatment E[Y|X,D=1] 
    g : np.ndarray
        Propensity scores P(D=1|X)
    theta_hat : float
        Estimated treatment effect
        
    Returns
    -------
    np.ndarray
        AIPW scores for each observation
    """
    return (m1 - m0) + d * (y - m1) / g - (1 - d) * (y - m0) / (1 - g) - theta_hat


def oos_moment_check(fold_thetas: List[float], fold_indices: List[np.ndarray], 
                    y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, g: np.ndarray) -> Tuple[pd.DataFrame, float]:
    """
    Out-of-sample moment check to avoid tautological results.
    
    For each fold k, evaluates the AIPW score using θ fitted on other folds,
    then tests if the combined moment condition holds.
    
    Parameters
    ----------
    fold_thetas : List[float]
        Treatment effects estimated excluding each fold
    fold_indices : List[np.ndarray] 
        Indices for each fold
    y, d, m0, m1, g : np.ndarray
        Data arrays (outcomes, treatment, predictions)
        
    Returns
    -------
    Tuple[pd.DataFrame, float]
        Fold-wise results and combined t-statistic
    """
    rows = []
    for k, idx in enumerate(fold_indices):
        th = fold_thetas[k]
        psi = aipw_score(y[idx], d[idx], m0[idx], m1[idx], g[idx], th)
        rows.append({"fold": k, "n": len(idx), "psi_mean": psi.mean(), "psi_var": psi.var(ddof=1)})
    
    df = pd.DataFrame(rows)
    num = (df["n"] * df["psi_mean"]).sum()
    den = np.sqrt((df["n"] * df["psi_var"]).sum())
    tstat = num / den if den > 0 else 0.0
    return df, float(tstat)


def orthogonality_derivatives(X_basis: np.ndarray, y: np.ndarray, d: np.ndarray, 
                            m0: np.ndarray, m1: np.ndarray, g: np.ndarray) -> pd.DataFrame:
    """
    Compute orthogonality (Gateaux derivative) tests for nuisance functions.
    
    Tests directional derivatives of the AIPW signal with respect to nuisances.
    For true nuisances, these derivatives should be ≈ 0 for rich sets of directions.
    
    Parameters
    ----------
    X_basis : np.ndarray, shape (n, B)
        Matrix of direction functions evaluated at X (include column of 1s for calibration)
    y, d, m0, m1, g : np.ndarray
        Data arrays
        
    Returns
    -------
    pd.DataFrame
        Derivative estimates, standard errors, and t-statistics for each basis function
    """
    n, B = X_basis.shape
    
    # m1 direction: ∂_{m1} φ̄[h1] = (1/n)∑ h1(Xi)(1 - Di/gi)
    dm1_terms = X_basis * (1 - d / g)[:, None]
    dm1 = dm1_terms.mean(axis=0)
    dm1_se = dm1_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    # m0 direction: ∂_{m0} φ̄[h0] = (1/n)∑ h0(Xi)((1-Di)/(1-gi) - 1)
    dm0_terms = X_basis * ((1 - d) / (1 - g) - 1)[:, None]
    dm0 = dm0_terms.mean(axis=0)  
    dm0_se = dm0_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    # g direction: ∂_g φ̄[s] = -(1/n)∑ s(Xi)[Di(Yi-m1i)/gi² + (1-Di)(Yi-m0i)/(1-gi)²]
    g_summand = (d * (y - m1) / g**2 + (1 - d) * (y - m0) / (1 - g)**2)
    dg_terms = -X_basis * g_summand[:, None]
    dg = dg_terms.mean(axis=0)
    dg_se = dg_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    out = pd.DataFrame({
        "basis": np.arange(B),
        "d_m1": dm1, "se_m1": dm1_se, "t_m1": dm1 / np.maximum(dm1_se, 1e-12),
        "d_m0": dm0, "se_m0": dm0_se, "t_m0": dm0 / np.maximum(dm0_se, 1e-12),
        "d_g": dg, "se_g": dg_se, "t_g": dg / np.maximum(dg_se, 1e-12),
    })
    return out


def influence_summary(y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, 
                     g: np.ndarray, theta_hat: float, k: int = 10) -> Dict[str, Any]:
    """
    Compute influence diagnostics showing where uncertainty comes from.
    
    Parameters
    ----------
    y, d, m0, m1, g : np.ndarray
        Data arrays
    theta_hat : float
        Estimated treatment effect
    k : int, default 10
        Number of top influential observations to return
        
    Returns
    -------
    Dict[str, Any]
        Influence diagnostics including SE, heavy-tail metrics, and top-k cases
    """
    psi = aipw_score(y, d, m0, m1, g, theta_hat)
    se = psi.std(ddof=1) / np.sqrt(len(psi))
    idx = np.argsort(-np.abs(psi))[:k]
    
    top = pd.DataFrame({
        "i": idx, 
        "psi": psi[idx], 
        "g": g[idx],
        "res_t": d[idx] * (y[idx] - m1[idx]),
        "res_c": (1 - d[idx]) * (y[idx] - m0[idx])
    })
    
    return {
        "se_plugin": float(se),
        "kurtosis": float(((psi - psi.mean())**4).mean() / (psi.var(ddof=1)**2 + 1e-12)),
        "p99_over_med": float(np.quantile(np.abs(psi), 0.99) / (np.median(np.abs(psi)) + 1e-12)),
        "top_influential": top
    }


def refute_irm_orthogonality(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    trim_propensity: Tuple[float, float] = (0.02, 0.98),
    n_basis_funcs: int = 3,
    n_folds_oos: int = 5,
    **inference_kwargs,
) -> Dict[str, Any]:
    """
    Comprehensive AIPW orthogonality diagnostics for DoubleML estimators.
    
    Implements three key diagnostic approaches based on the efficient influence function (EIF):
    1. Out-of-sample moment check (non-tautological)
    2. Orthogonality (Gateaux derivative) tests  
    3. Influence diagnostics
    
    Parameters
    ----------
    inference_fn : Callable
        The inference function (dml_ate or dml_att)
    data : CausalData
        The causal data object
    trim_propensity : Tuple[float, float], default (0.02, 0.98)
        Propensity score trimming bounds (min, max) to avoid extreme weights
    n_basis_funcs : int, default 3
        Number of basis functions for orthogonality derivative tests (constant + covariates)
    n_folds_oos : int, default 5
        Number of folds for out-of-sample moment check
    **inference_kwargs : dict
        Additional arguments passed to inference_fn
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - oos_moment_test: Out-of-sample moment condition results
        - orthogonality_derivatives: Gateaux derivative test results
        - influence_diagnostics: Influence function diagnostics
        - theta: Original treatment effect estimate
        - trimmed_diagnostics: Results on trimmed sample
        - overall_assessment: Summary diagnostic assessment
        
    Examples
    --------
    >>> from causalkit.refutation import refute_irm_orthogonality
    >>> from causalkit.inference.ate import dml_ate
    >>> 
    >>> # Comprehensive orthogonality check
    >>> ortho_results = refute_irm_orthogonality(dml_ate, causal_data)
    >>> 
    >>> # Check key diagnostics
    >>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
    >>> print(f"Calibration issues: {len(ortho_results['orthogonality_derivatives'].query('abs(t_g) > 2'))}")
    >>> print(f"Assessment: {ortho_results['overall_assessment']}")
    """
    # Run inference to get fitted DoubleML model
    result = inference_fn(data, **inference_kwargs)
    
    if 'model' not in result:
        raise ValueError("Inference function must return a dictionary with 'model' key")
    
    dml_model = result['model']
    theta_hat = result['coefficient']
    
    # Extract cross-fitted predictions from DoubleML model
    try:
        g_propensity = dml_model.predictions['ml_m'][:, 0]  # P(D=1|X)
        
        # Handle different outcome prediction structures
        if hasattr(dml_model, '_dml_data') and hasattr(dml_model._dml_data, 'd_cols'):
            # For IRM, DoubleML fits separate models for E[Y|X,D=0] and E[Y|X,D=1]
            if 'ml_g0' in dml_model.predictions and 'ml_g1' in dml_model.predictions:
                m0_outcomes = dml_model.predictions['ml_g0'][:, 0]
                m1_outcomes = dml_model.predictions['ml_g1'][:, 0]
            else:
                # Fallback: use combined prediction (less precise but workable)
                combined_pred = dml_model.predictions['ml_g'][:, 0]
                m0_outcomes = combined_pred
                m1_outcomes = combined_pred
        else:
            raise ValueError("Could not extract outcome predictions from DoubleML model")
            
    except (KeyError, IndexError, AttributeError) as e:
        raise ValueError(f"Error extracting predictions from DoubleML model: {e}")
    
    # Get observed data
    y = data.target.values.astype(float)
    d = data.treatment.values.astype(float)
    
    # Ensure all prediction arrays are 1D to match y and d
    g_propensity = np.asarray(g_propensity).flatten()
    m0_outcomes = np.asarray(m0_outcomes).flatten()
    m1_outcomes = np.asarray(m1_outcomes).flatten()
    
    # Apply propensity trimming
    trim_min, trim_max = trim_propensity
    trim_mask = (g_propensity >= trim_min) & (g_propensity <= trim_max)
    n_trimmed = np.sum(~trim_mask)
    
    # Create trimmed arrays
    y_trim = y[trim_mask]
    d_trim = d[trim_mask]
    g_trim = g_propensity[trim_mask]
    m0_trim = m0_outcomes[trim_mask]
    m1_trim = m1_outcomes[trim_mask]
    
    # === 1. OUT-OF-SAMPLE MOMENT CHECK ===
    # Refit model on K-1 folds, evaluate on held-out fold
    kf = KFold(n_splits=n_folds_oos, shuffle=True, random_state=42)
    fold_thetas = []
    fold_indices = []
    
    df_full = data.get_df()
    
    for train_idx, test_idx in kf.split(df_full):
        # Create fold-specific data
        train_data = CausalData(
            df=df_full.iloc[train_idx].copy(),
            treatment=data._treatment,
            outcome=data._target,
            cofounders=data._cofounders
        )
        
        # Fit model on training fold
        fold_result = inference_fn(train_data, **inference_kwargs)
        fold_thetas.append(fold_result['coefficient'])
        fold_indices.append(test_idx)
    
    # Run out-of-sample moment check
    oos_df, oos_tstat = oos_moment_check(
        fold_thetas, fold_indices, y, d, m0_outcomes, m1_outcomes, g_propensity
    )
    oos_pvalue = 2 * (1 - stats.norm.cdf(np.abs(oos_tstat)))
    
    # === 2. ORTHOGONALITY DERIVATIVE TESTS ===
    # Create basis functions: constant + first few covariates
    X = data.get_df()[data._cofounders].values
    n_covs = min(n_basis_funcs - 1, X.shape[1])  # -1 for constant term
    X_basis = np.c_[np.ones(len(X)), X[:, :n_covs]]  # Constant + first n_covs covariates
    
    # Run on full sample
    ortho_derivs_full = orthogonality_derivatives(X_basis, y, d, m0_outcomes, m1_outcomes, g_propensity)
    
    # Run on trimmed sample
    X_basis_trim = X_basis[trim_mask]
    ortho_derivs_trim = orthogonality_derivatives(X_basis_trim, y_trim, d_trim, m0_trim, m1_trim, g_trim)
    
    # === 3. INFLUENCE DIAGNOSTICS ===
    # Full sample
    influence_full = influence_summary(y, d, m0_outcomes, m1_outcomes, g_propensity, theta_hat)
    
    # Trimmed sample
    influence_trim = influence_summary(y_trim, d_trim, m0_trim, m1_trim, g_trim, theta_hat)
    
    # === DIAGNOSTIC ASSESSMENT ===
    # Check for problematic derivatives (|t-stat| > 2)
    problematic_derivs_full = ortho_derivs_full[(np.abs(ortho_derivs_full['t_m1']) > 2) | 
                                               (np.abs(ortho_derivs_full['t_m0']) > 2) | 
                                               (np.abs(ortho_derivs_full['t_g']) > 2)]
    
    problematic_derivs_trim = ortho_derivs_trim[(np.abs(ortho_derivs_trim['t_m1']) > 2) | 
                                               (np.abs(ortho_derivs_trim['t_m0']) > 2) | 
                                               (np.abs(ortho_derivs_trim['t_g']) > 2)]
    
    # Overall assessment
    conditions = {
        'oos_moment_ok': abs(oos_tstat) < 2.0,
        'derivs_full_ok': len(problematic_derivs_full) == 0,
        'derivs_trim_ok': len(problematic_derivs_trim) == 0,
        'se_reasonable': influence_full['se_plugin'] < 10 * result.get('std_error', influence_full['se_plugin']),
        'no_extreme_influence': influence_full['p99_over_med'] < 10,
        'trimming_reasonable': n_trimmed < 0.1 * len(y)
    }
    
    n_passed = sum(conditions.values())
    if n_passed >= 5:
        overall_assessment = "PASS: Strong evidence for orthogonality"
    elif n_passed >= 4:
        overall_assessment = "CAUTION: Most conditions satisfied" 
    elif n_passed >= 3:
        overall_assessment = "WARNING: Several orthogonality violations"
    else:
        overall_assessment = "FAIL: Major orthogonality violations detected"
    
    return {
        'theta': float(theta_hat),
        'oos_moment_test': {
            'fold_results': oos_df,
            'tstat': float(oos_tstat),
            'pvalue': float(oos_pvalue),
            'interpretation': 'Should be ≈ 0 if moment condition holds'
        },
        'orthogonality_derivatives': {
            'full_sample': ortho_derivs_full,
            'trimmed_sample': ortho_derivs_trim,
            'problematic_full': problematic_derivs_full,
            'problematic_trimmed': problematic_derivs_trim,
            'interpretation': 'Large |t-stats| (>2) indicate calibration issues'
        },
        'influence_diagnostics': {
            'full_sample': influence_full,
            'trimmed_sample': influence_trim,
            'interpretation': 'Heavy tails or extreme kurtosis suggest instability'
        },
        'trimming_info': {
            'bounds': trim_propensity,
            'n_trimmed': int(n_trimmed),
            'pct_trimmed': float(n_trimmed / len(y) * 100)
        },
        'diagnostic_conditions': conditions,
        'overall_assessment': overall_assessment
    }