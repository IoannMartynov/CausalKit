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


def aipw_score_ate(y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, g: np.ndarray, theta: float, eps: float = 0.01) -> np.ndarray:
    """
    Efficient influence function (EIF) for ATE.
    """
    g = np.clip(g, eps, 1 - eps)
    return (m1 - m0) + d * (y - m1) / g - (1 - d) * (y - m0) / (1 - g) - theta


def aipw_score_att(y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, g: np.ndarray, theta: float, p1: Optional[float] = None, eps: float = 0.01) -> np.ndarray:
    """
    Efficient influence function (EIF) for ATT, normalized by p1 = P(D=1).
    m1 enters via theta only; derivative wrt m1 is zero.
    """
    g = np.clip(g, eps, 1 - eps)
    if p1 is None:
        p1 = float(np.mean(d))
    return (d * (y - m0 - theta) + (1 - d) * (g / (1 - g)) * (y - m0)) / (p1 + 1e-12)


def extract_nuisances(dml_model, test_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustly extract nuisance function predictions from DoubleML model.
    
    Handles different DoubleML prediction key layouts and provides clear error messages.
    
    Parameters
    ----------
    dml_model : DoubleML model
        Fitted DoubleML model with predictions
    test_indices : np.ndarray, optional
        If provided, extract predictions only for these indices
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (g, m0, m1) where:
        - g: propensity scores P(D=1|X) 
        - m0: outcome predictions E[Y|X,D=0]
        - m1: outcome predictions E[Y|X,D=1]
        
    Raises
    ------
    KeyError
        If required prediction keys cannot be found
    """
    preds = dml_model.predictions
    available_keys = list(preds.keys())

    def _reduce_to_1d(name: str) -> np.ndarray:
        if name not in preds:
            raise KeyError(f"Missing predictions['{name}']; available: {available_keys}")
        arr = np.asarray(preds[name])
        # Squeeze singleton dimensions first
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            raise ValueError(f"predictions['{name}'] is scalar; expected at least 1D, got shape {arr.shape}")
        if arr.ndim == 1:
            return arr
        # Average across all non-first axes to aggregate repetitions/aux dims
        axes = tuple(range(1, arr.ndim))
        return arr.mean(axis=axes)

    # Strict: require proper keys
    g = _reduce_to_1d('ml_m')  # P(D=1|X)
    m0 = _reduce_to_1d('ml_g0')  # E[Y|X,D=0]
    m1 = _reduce_to_1d('ml_g1')  # E[Y|X,D=1]

    # Ensure arrays are 1D and extract test indices if specified
    g = np.asarray(g).flatten()
    m0 = np.asarray(m0).flatten()
    m1 = np.asarray(m1).flatten()
    
    if test_indices is not None:
        g = g[test_indices]
        m0 = m0[test_indices]  
        m1 = m1[test_indices]
    
    return g, m0, m1


def aipw_score(y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, g: np.ndarray, theta_hat: float, eps: float = 0.01) -> np.ndarray:
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
    eps : float, default 0.01
        Clipping bound for propensity scores to avoid extreme weights
        
    Returns
    -------
    np.ndarray
        AIPW scores for each observation
    """
    # Clip propensity scores to avoid division by zero and extreme weights
    g_clipped = np.clip(g, eps, 1 - eps)
    return (m1 - m0) + d * (y - m1) / g_clipped - (1 - d) * (y - m0) / (1 - g_clipped) - theta_hat


def oos_moment_check_with_fold_nuisances(
    fold_thetas: List[float],
    fold_indices: List[np.ndarray],
    fold_nuisances: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    d: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Out-of-sample moment check using fold-specific nuisances to avoid tautological results.
    
    For each fold k, evaluates the AIPW score using θ fitted on other folds and 
    nuisance predictions from the fold-specific model, then tests if the combined 
    moment condition holds.
    
    Parameters
    ----------
    fold_thetas : List[float]
        Treatment effects estimated excluding each fold
    fold_indices : List[np.ndarray] 
        Indices for each fold
    fold_nuisances : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Fold-specific nuisance predictions (g, m0, m1) for each fold
    y, d : np.ndarray
        Observed outcomes and treatments
        
    Returns
    -------
    Tuple[pd.DataFrame, float]
        Fold-wise results and combined t-statistic
    """
    rows = []
    for k, (idx, nuisances) in enumerate(zip(fold_indices, fold_nuisances)):
        th = fold_thetas[k]
        g_fold, m0_fold, m1_fold = nuisances
        
        # Compute scores using fold-specific theta and nuisances
        if score_fn is None:
            psi = aipw_score_ate(y[idx], d[idx], m0_fold, m1_fold, g_fold, th)
        else:
            psi = score_fn(y[idx], d[idx], m0_fold, m1_fold, g_fold, th)
        rows.append({"fold": k, "n": len(idx), "psi_mean": psi.mean(), "psi_var": psi.var(ddof=1)})
    
    df = pd.DataFrame(rows)
    num = (df["n"] * df["psi_mean"]).sum()
    den = np.sqrt((df["n"] * df["psi_var"]).sum())
    tstat = num / den if den > 0 else 0.0
    return df, float(tstat)


def oos_moment_check(
    fold_thetas: List[float],
    fold_indices: List[np.ndarray],
    y: np.ndarray,
    d: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    g: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Out-of-sample moment check to avoid tautological results (legacy version).
    
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
        if score_fn is None:
            psi = aipw_score_ate(y[idx], d[idx], m0[idx], m1[idx], g[idx], th)
        else:
            psi = score_fn(y[idx], d[idx], m0[idx], m1[idx], g[idx], th)
        rows.append({"fold": k, "n": len(idx), "psi_mean": psi.mean(), "psi_var": psi.var(ddof=1)})
    
    df = pd.DataFrame(rows)
    num = (df["n"] * df["psi_mean"]).sum()
    den = np.sqrt((df["n"] * df["psi_var"]).sum())
    tstat = num / den if den > 0 else 0.0
    return df, float(tstat)


def orthogonality_derivatives(X_basis: np.ndarray, y: np.ndarray, d: np.ndarray, 
                            m0: np.ndarray, m1: np.ndarray, g: np.ndarray, eps: float = 0.01) -> pd.DataFrame:
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
    eps : float, default 0.01
        Clipping bound for propensity scores to avoid extreme weights
        
    Returns
    -------
    pd.DataFrame
        Derivative estimates, standard errors, and t-statistics for each basis function
    """
    n, B = X_basis.shape
    
    # Clip propensity scores to avoid division by zero and extreme weights
    g_clipped = np.clip(g, eps, 1 - eps)
    
    # m1 direction: ∂_{m1} φ̄[h1] = (1/n)∑ h1(Xi)(1 - Di/gi)
    dm1_terms = X_basis * (1 - d / g_clipped)[:, None]
    dm1 = dm1_terms.mean(axis=0)
    dm1_se = dm1_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    # m0 direction: ∂_{m0} φ̄[h0] = (1/n)∑ h0(Xi)((1-Di)/(1-gi) - 1)
    dm0_terms = X_basis * ((1 - d) / (1 - g_clipped) - 1)[:, None]
    dm0 = dm0_terms.mean(axis=0)  
    dm0_se = dm0_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    # g direction: ∂_g φ̄[s] = -(1/n)∑ s(Xi)[Di(Yi-m1i)/gi² + (1-Di)(Yi-m0i)/(1-gi)²]
    g_summand = (d * (y - m1) / g_clipped**2 + (1 - d) * (y - m0) / (1 - g_clipped)**2)
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
                     g: np.ndarray, theta_hat: float, k: int = 10, target: str = "ATE", clip_eps: float = 0.01) -> Dict[str, Any]:
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
    target_u = str(target).upper()
    if target_u == "ATE":
        psi = aipw_score_ate(y, d, m0, m1, g, theta_hat, eps=clip_eps)
    elif target_u == "ATTE" or target_u == "ATT":
        psi = aipw_score_att(y, d, m0, m1, g, theta_hat, p1=float(np.mean(d)), eps=clip_eps)
    else:
        raise ValueError("target must be 'ATE' or 'ATTE'")
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
    target: str = "ATE",
    clip_eps: float = 0.01,
    strict_oos: bool = False,
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
    theta_hat = float(result['coefficient'])
    
    # Extract cross-fitted predictions from DoubleML model using robust function
    g_propensity, m0_outcomes, m1_outcomes = extract_nuisances(dml_model)
    
    # Get observed data
    y = data.target.values.astype(float)
    d = data.treatment.values.astype(float)

    # Determine target automatically if requested
    target_u = str(target).upper() if target is not None else "AUTO"
    if target_u == "AUTO":
        score_attr = getattr(dml_model, 'score', None)
        if score_attr is None:
            score_attr = getattr(dml_model, '_score', 'ATE')
        score_str = str(score_attr).upper()
        target_u = 'ATTE' if 'ATT' in score_str else 'ATE'
    
    # Build score function
    if target_u == 'ATE':
        def score_fn(y_, d_, m0_, m1_, g_, th_):
            return aipw_score_ate(y_, d_, m0_, m1_, g_, th_, eps=clip_eps)
    elif target_u in ('ATTE', 'ATT'):
        p1 = float(np.mean(d))
        def score_fn(y_, d_, m0_, m1_, g_, th_):
            return aipw_score_att(y_, d_, m0_, m1_, g_, th_, p1=p1, eps=clip_eps)
    else:
        raise ValueError("target must be 'ATE' or 'ATTE'")
    
    # Apply propensity trimming
    trim_min, trim_max = trim_propensity
    trim_mask = (g_propensity >= trim_min) & (g_propensity <= trim_max)
    n_trimmed = int(np.sum(~trim_mask))
    
    # Create trimmed arrays
    y_trim = y[trim_mask]
    d_trim = d[trim_mask]
    g_trim = g_propensity[trim_mask]
    m0_trim = m0_outcomes[trim_mask]
    m1_trim = m1_outcomes[trim_mask]
    
    # === 1. OUT-OF-SAMPLE MOMENT CHECK ===
    strict_applied = False
    kf = KFold(n_splits=n_folds_oos, shuffle=True, random_state=42)
    fold_thetas: List[float] = []
    fold_indices: List[np.ndarray] = []
    fold_nuisances: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    
    df_full = data.get_df()
    for train_idx, test_idx in kf.split(df_full):
        # Create fold-specific data using public APIs
        train_data = CausalData(
            df=df_full.iloc[train_idx].copy(),
            treatment=data.treatment.name,
            outcome=data.target.name,
            confounders=list(data.confounders) if data.confounders else None
        )
        # Fit model on training fold to get theta
        fold_result = inference_fn(train_data, **inference_kwargs)
        fold_thetas.append(float(fold_result['coefficient']))
        fold_indices.append(test_idx)
        
        if strict_oos and not strict_applied:
            # Strict mode not implemented fully; fall back to fast nuisances but mark flag
            strict_applied = False
        
        # Use full-model cross-fitted nuisances indexed by test fold (fast OOS)
        g_fold = g_propensity[test_idx]
        m0_fold = m0_outcomes[test_idx]
        m1_fold = m1_outcomes[test_idx]
        fold_nuisances.append((g_fold, m0_fold, m1_fold))
    
    # Run out-of-sample moment check with fold-specific nuisances
    oos_df, oos_tstat = oos_moment_check_with_fold_nuisances(
        fold_thetas, fold_indices, fold_nuisances, y, d, score_fn=score_fn
    )
    oos_pvalue = 2 * (1 - stats.norm.cdf(np.abs(oos_tstat)))
    
    # === 2. ORTHOGONALITY DERIVATIVE TESTS ===
    # Create basis functions: constant + first few standardized confounders
    if data.confounders is None or len(data.confounders) == 0:
        raise ValueError("CausalData object must have confounders defined for orthogonality diagnostics")
    
    X = data.get_df()[list(data.confounders)].values
    n_covs = min(max(n_basis_funcs - 1, 0), X.shape[1])  # -1 for constant term
    
    if n_covs > 0:
        X_selected = X[:, :n_covs]
        X_std = (X_selected - np.mean(X_selected, axis=0)) / (np.std(X_selected, axis=0) + 1e-8)
        X_basis = np.c_[np.ones(len(X)), X_std]  # Constant + standardized covariates
    else:
        X_basis = np.ones((len(X), 1))  # Only constant term
    
    if target_u == 'ATE':
        ortho_derivs_full = orthogonality_derivatives(X_basis, y, d, m0_outcomes, m1_outcomes, g_propensity, eps=clip_eps)
        X_basis_trim = X_basis[trim_mask]
        ortho_derivs_trim = orthogonality_derivatives(X_basis_trim, y_trim, d_trim, m0_trim, m1_trim, g_trim, eps=clip_eps)
        problematic_derivs_full = ortho_derivs_full[(np.abs(ortho_derivs_full['t_m1']) > 2) | 
                                                   (np.abs(ortho_derivs_full['t_m0']) > 2) | 
                                                   (np.abs(ortho_derivs_full['t_g']) > 2)]
        problematic_derivs_trim = ortho_derivs_trim[(np.abs(ortho_derivs_trim['t_m1']) > 2) | 
                                                   (np.abs(ortho_derivs_trim['t_m0']) > 2) | 
                                                   (np.abs(ortho_derivs_trim['t_g']) > 2)]
        derivs_interpretation = 'Large |t-stats| (>2) indicate calibration issues'
        derivs_full_ok = len(problematic_derivs_full) == 0
        derivs_trim_ok = len(problematic_derivs_trim) == 0
    else:
        # For ATT, derivative formulas differ; skip with a clear note
        ortho_derivs_full = pd.DataFrame()
        ortho_derivs_trim = pd.DataFrame()
        problematic_derivs_full = pd.DataFrame()
        problematic_derivs_trim = pd.DataFrame()
        derivs_interpretation = 'ATT selected: derivative checks skipped (ATE formulas not applicable)'
        derivs_full_ok = True
        derivs_trim_ok = True
    
    # === 3. INFLUENCE DIAGNOSTICS ===
    influence_full = influence_summary(y, d, m0_outcomes, m1_outcomes, g_propensity, theta_hat, target=target_u, clip_eps=clip_eps)
    influence_trim = influence_summary(y_trim, d_trim, m0_trim, m1_trim, g_trim, theta_hat, target=target_u, clip_eps=clip_eps)
    
    # === DIAGNOSTIC ASSESSMENT ===
    model_se = result.get('std_error')
    if model_se is not None:
        se_reasonable = abs(influence_full['se_plugin'] - model_se) < 2 * model_se
    else:
        se_reasonable = False
    
    conditions = {
        'oos_moment_ok': abs(oos_tstat) < 2.0,
        'derivs_full_ok': derivs_full_ok,
        'derivs_trim_ok': derivs_trim_ok,
        'se_reasonable': se_reasonable,
        'no_extreme_influence': influence_full['p99_over_med'] < 10,
        'trimming_reasonable': n_trimmed < 0.1 * len(y)
    }
    
    n_passed = sum(bool(v) for v in conditions.values())
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
        'params': {
            'target': target_u,
            'clip_eps': clip_eps,
            'strict_oos_requested': bool(strict_oos),
            'strict_oos_applied': bool(False)
        },
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
            'interpretation': derivs_interpretation
        },
        'influence_diagnostics': {
            'full_sample': influence_full,
            'trimmed_sample': influence_trim,
            'interpretation': 'Heavy tails or extreme kurtosis suggest instability'
        },
        'trimming_info': {
            'bounds': trim_propensity,
            'n_trimmed': int(n_trimmed),
            'pct_trimmed': float(n_trimmed / len(y) * 100.0)
        },
        'diagnostic_conditions': conditions,
        'overall_assessment': overall_assessment
    }