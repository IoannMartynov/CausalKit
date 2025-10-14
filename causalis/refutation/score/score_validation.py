"""
AIPW orthogonality diagnostics for IRM-based estimators.

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

from causalis.data.causaldata import CausalData
from copy import deepcopy


def aipw_score_ate(y: np.ndarray, d: np.ndarray, g0: np.ndarray, g1: np.ndarray, m: np.ndarray, theta: float, trimming_threshold: float = 0.01) -> np.ndarray:
    """
    Efficient influence function (EIF) for ATE.
    Uses IRM naming: g0,g1 are outcome regressions E[Y|X,D=0/1], m is propensity P(D=1|X).
    """
    m = np.clip(m, trimming_threshold, 1 - trimming_threshold)
    return (g1 - g0) + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m) - theta


def aipw_score_atte(y: np.ndarray, d: np.ndarray, g0: np.ndarray, g1: np.ndarray, m: np.ndarray, theta: float, p_treated: Optional[float] = None, trimming_threshold: float = 0.01) -> np.ndarray:
    """
    Efficient influence function (EIF) for ATTE under IRM/AIPW.

    ψ_ATTE(W; θ, η) = [ D*(Y - g0(X) - θ)  -  (1-D)*{ m(X)/(1-m(X)) }*(Y - g0(X)) ] / E[D]

    Notes:
      - Matches DoubleML's `score='ATTE'` (weights ω=D/E[D], \bar{ω}=m(X)/E[D]).
      - g1 enters only via θ; ∂ψ/∂g1 = 0.
    """
    m = np.clip(m, trimming_threshold, 1 - trimming_threshold)
    if p_treated is None:
        p_treated = float(np.mean(d))
    gamma = m / (1.0 - m)
    num = d * (y - g0 - theta) - (1.0 - d) * gamma * (y - g0)
    return num / (p_treated + 1e-12)




def extract_nuisances(model, test_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.

    Tries several backends for robustness:
      1) IRM attributes: m_hat_, g0_hat_, g1_hat_
      2) model.predictions dict with keys: 'ml_m','ml_g0','ml_g1'
      3) Direct attributes: ml_m, ml_g0, ml_g1

    Parameters
    ----------
    model : object
        Fitted internal IRM estimator (causalis.inference.estimators.IRM) or a compatible dummy model
    test_indices : np.ndarray, optional
        If provided, extract predictions only for these indices

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (m, g0, g1) where:
        - m: propensity scores P(D=1|X)
        - g0: outcome predictions E[Y|X,D=0]
        - g1: outcome predictions E[Y|X,D=1]
    """
    m = g0 = g1 = None

    # 1) Preferred IRM attributes
    if hasattr(model, "m_hat_") and getattr(model, "m_hat_", None) is not None:
        m = np.asarray(getattr(model, "m_hat_"), dtype=float).ravel()
        g0 = np.asarray(getattr(model, "g0_hat_"), dtype=float).ravel()
        g1 = np.asarray(getattr(model, "g1_hat_"), dtype=float).ravel()
    else:
        # 2) Predictions dict backend
        preds = getattr(model, "predictions", None)
        if isinstance(preds, dict) and all(k in preds for k in ("ml_m", "ml_g0", "ml_g1")):
            m = np.asarray(preds["ml_m"], dtype=float).ravel()
            g0 = np.asarray(preds["ml_g0"], dtype=float).ravel()
            g1 = np.asarray(preds["ml_g1"], dtype=float).ravel()
        else:
            # 3) Direct attributes with same names
            if all(hasattr(model, nm) for nm in ("ml_m", "ml_g0", "ml_g1")):
                m = np.asarray(getattr(model, "ml_m"), dtype=float).ravel()
                g0 = np.asarray(getattr(model, "ml_g0"), dtype=float).ravel()
                g1 = np.asarray(getattr(model, "ml_g1"), dtype=float).ravel()

    if m is None or g0 is None or g1 is None:
        raise AttributeError("IRM model-compatible nuisances not found. Expected m_hat_/g0_hat_/g1_hat_ or predictions['ml_*'].")

    if test_indices is not None:
        m = m[test_indices]
        g0 = g0[test_indices]
        g1 = g1[test_indices]

    return m, g0, g1




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
        Fold-specific nuisance predictions (m, g0, g1) for each fold
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
        m_fold, g0_fold, g1_fold = nuisances
        
        # Compute scores using fold-specific theta and nuisances
        if score_fn is None:
            psi = aipw_score_ate(y[idx], d[idx], g0_fold, g1_fold, m_fold, th)
        else:
            psi = score_fn(y[idx], d[idx], g0_fold, g1_fold, m_fold, th)
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
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    score_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Out-of-sample moment check to avoid tautological results (legacy/simple version).
    
    For each fold k, evaluates the AIPW score using θ fitted on other folds,
    then tests if the combined moment condition holds.
    
    Parameters
    ----------
    fold_thetas : List[float]
        Treatment effects estimated excluding each fold
    fold_indices : List[np.ndarray] 
        Indices for each fold
    y, d, g0, g1, m : np.ndarray
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
            psi = aipw_score_ate(y[idx], d[idx], g0[idx], g1[idx], m[idx], th)
        else:
            psi = score_fn(y[idx], d[idx], g0[idx], g1[idx], m[idx], th)
        rows.append({"fold": k, "n": len(idx), "psi_mean": psi.mean(), "psi_var": psi.var(ddof=1)})
    
    df = pd.DataFrame(rows)
    num = (df["n"] * df["psi_mean"]).sum()
    den = np.sqrt((df["n"] * df["psi_var"]).sum())
    tstat = num / den if den > 0 else 0.0
    return df, float(tstat)


def orthogonality_derivatives(X_basis: np.ndarray, y: np.ndarray, d: np.ndarray, 
                            g0: np.ndarray, g1: np.ndarray, m: np.ndarray, trimming_threshold: float = 0.01) -> pd.DataFrame:
    """
    Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
    Uses IRM naming: g0,g1 outcomes; m propensity.
    """
    n, B = X_basis.shape
    
    # Clip propensity scores to avoid division by zero and extreme weights
    m_clipped = np.clip(m, trimming_threshold, 1 - trimming_threshold)
    
    # g1 direction: ∂_{g1} φ̄[h1] = (1/n)∑ h1(Xi)(1 - Di/mi)
    dg1_terms = X_basis * (1 - d / m_clipped)[:, None]
    dg1 = dg1_terms.mean(axis=0)
    dg1_se = dg1_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    # g0 direction: ∂_{g0} φ̄[h0] = (1/n)∑ h0(Xi)((1-Di)/(1-mi) - 1)
    dg0_terms = X_basis * ((1 - d) / (1 - m_clipped) - 1)[:, None]
    dg0 = dg0_terms.mean(axis=0)  
    dg0_se = dg0_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    # m direction: ∂_m φ̄[s] = -(1/n)∑ s(Xi)[Di(Yi-g1i)/mi² + (1-Di)(Yi-g0i)/(1-mi)²]
    m_summand = (d * (y - g1) / m_clipped**2 + (1 - d) * (y - g0) / (1 - m_clipped)**2)
    dm_terms = -X_basis * m_summand[:, None]
    dm = dm_terms.mean(axis=0)
    dm_se = dm_terms.std(axis=0, ddof=1) / np.sqrt(n)
    
    out = pd.DataFrame({
        "basis": np.arange(B),
        "d_g1": dg1, "se_g1": dg1_se, "t_g1": dg1 / np.maximum(dg1_se, 1e-12),
        "d_g0": dg0, "se_g0": dg0_se, "t_g0": dg0 / np.maximum(dg0_se, 1e-12),
        "d_m": dm,  "se_m": dm_se,  "t_m":  dm / np.maximum(dm_se, 1e-12),
    })
    return out


def influence_summary(y: np.ndarray, d: np.ndarray, g0: np.ndarray, g1: np.ndarray, 
                     m: np.ndarray, theta_hat: float, k: int = 10, score: str = "ATE", trimming_threshold: float = 0.01) -> Dict[str, Any]:
    """
    Compute influence diagnostics showing where uncertainty comes from.
    
    Parameters
    ----------
    y, d, g0, g1, m : np.ndarray
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
    score_u = str(score).upper()
    if score_u == "ATE":
        psi = aipw_score_ate(y, d, g0, g1, m, theta_hat, trimming_threshold=trimming_threshold)
    elif score_u in ("ATTE", "ATT"):
        psi = aipw_score_atte(y, d, g0, g1, m, theta_hat, p_treated=float(np.mean(d)), trimming_threshold=trimming_threshold)
    else:
        raise ValueError("score must be 'ATE' or 'ATTE'")
    se = psi.std(ddof=1) / np.sqrt(len(psi))
    idx = np.argsort(-np.abs(psi))[:k]
    
    top = pd.DataFrame({
        "i": idx, 
        "psi": psi[idx], 
        "m": m[idx],
        "res_t": d[idx] * (y[idx] - g1[idx]),
        "res_c": (1 - d[idx]) * (y[idx] - g0[idx])
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
    n_basis_funcs: Optional[int] = None,
    n_folds_oos: int = 4,
    score: Optional[str] = None,
    trimming_threshold: float = 0.01,
    strict_oos: bool = True,
    **inference_kwargs,
) -> Dict[str, Any]:
    """
    Comprehensive AIPW orthogonality diagnostics for IRM estimators.
    
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
    n_basis_funcs : Optional[int], default None (len(confounders)+1)
        Number of basis functions for orthogonality derivative tests (constant + covariates).
        If None, defaults to the number of confounders in `data` plus 1 for the constant term.
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
    >>> from causalis.refutation.orthogonality import refute_irm_orthogonality
    >>> from causalis.inference.ate.dml_ate import dml_ate
    >>> 
    >>> # Comprehensive orthogonality check
    >>> ortho_results = refute_irm_orthogonality(dml_ate, causal_data)
    >>> 
    >>> # Check key diagnostics
    >>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
    >>> print(f"Assessment: {ortho_results['overall_assessment']}")
    """
    # Run inference to get fitted IRM model
    result = inference_fn(data, **inference_kwargs)
    
    if 'model' not in result:
        raise ValueError("Inference function must return a dictionary with 'model' key")
    
    dml_model = result['model']
    # Prefer wrapper-reported coefficient; fallback to IRM attributes if missing
    if 'coefficient' in result:
        theta_hat = float(result['coefficient'])
    else:
        theta_hat = float(getattr(dml_model, 'coef_', [np.nan])[0])
    
    # Extract cross-fitted predictions from IRM model using robust function
    m_propensity, g0_outcomes, g1_outcomes = extract_nuisances(dml_model)
    
    # Get observed data
    y = data.target.values.astype(float)
    d = data.treatment.values.astype(float)

    # Determine score
    score_u = str(score).upper() if score is not None else "AUTO"
    if score_u == "AUTO":
        score_attr = getattr(dml_model, 'score', None)
        if score_attr is None:
            score_attr = getattr(dml_model, '_score', 'ATE')
        score_str = str(score_attr).upper()
        score_u = 'ATTE' if 'ATT' in score_str else 'ATE'
    
    # Build score function
    if score_u == 'ATE':
        def score_fn(y_, d_, g0_, g1_, m_, th_):
            return aipw_score_ate(y_, d_, g0_, g1_, m_, th_, trimming_threshold=trimming_threshold)
    elif score_u in ('ATTE', 'ATT'):
        p_treated = float(np.mean(d))
        def score_fn(y_, d_, g0_, g1_, m_, th_):
            return aipw_score_atte(y_, d_, g0_, g1_, m_, th_, p_treated=p_treated, trimming_threshold=trimming_threshold)
    else:
        raise ValueError("score must be 'ATE' or 'ATTE'")
    
    # Apply propensity trimming
    trim_min, trim_max = trim_propensity
    trim_mask = (m_propensity >= trim_min) & (m_propensity <= trim_max)
    n_trimmed = int(np.sum(~trim_mask))
    
    # Create trimmed arrays
    y_trim = y[trim_mask]
    d_trim = d[trim_mask]
    m_trim = m_propensity[trim_mask]
    g0_trim = g0_outcomes[trim_mask]
    g1_trim = g1_outcomes[trim_mask]
    
    # === 1. OUT-OF-SAMPLE MOMENT CHECK ===
    # Prefer fast path using cached ψ_a, ψ_b and training folds from the fitted IRM model
    use_fast = hasattr(dml_model, "psi_a_") and hasattr(dml_model, "psi_b_") and getattr(dml_model, "folds_", None) is not None
    if use_fast:
        folds_arr = np.asarray(dml_model.folds_, dtype=int)
        K = int(folds_arr.max() + 1) if folds_arr.size > 0 else 0
        fold_indices = [np.where(folds_arr == k)[0] for k in range(K)]
        # Compute both fold-aggregated and strict t-stats using only ψ components
        oos_df, oos_tstat, tstat_strict = oos_moment_check_from_psi(
            np.asarray(dml_model.psi_a_, dtype=float),
            np.asarray(dml_model.psi_b_, dtype=float),
            fold_indices,
            strict=True,
        )
        oos_pvalue = float(2 * (1 - stats.norm.cdf(abs(oos_tstat))))
        pvalue_strict = float(2 * (1 - stats.norm.cdf(abs(tstat_strict)))) if tstat_strict is not None else float("nan")
        strict_applied = bool(strict_oos)
        main_tstat = float(tstat_strict if strict_applied else oos_tstat)
        main_pvalue = float(2 * (1 - stats.norm.cdf(abs(main_tstat))))
    else:
        # Fallback: legacy slow path with K refits and score recomputations
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

            # Use full-model cross-fitted nuisances indexed by test fold (fast OOS)
            m_fold = m_propensity[test_idx]
            g0_fold = g0_outcomes[test_idx]
            g1_fold = g1_outcomes[test_idx]
            fold_nuisances.append((m_fold, g0_fold, g1_fold))

        # Run out-of-sample moment check with fold-specific nuisances (fold-aggregated)
        oos_df, oos_tstat = oos_moment_check_with_fold_nuisances(
            fold_thetas, fold_indices, fold_nuisances, y, d, score_fn=score_fn
        )
        oos_pvalue = float(2 * (1 - stats.norm.cdf(abs(oos_tstat))))

        # Strict aggregation over all held-out psi without creating a giant array
        total_n = 0
        total_sum = 0.0
        total_sumsq = 0.0
        for k, (idx, (m_fold, g0_fold, g1_fold)) in enumerate(zip(fold_indices, fold_nuisances)):
            th = fold_thetas[k]
            psi_k = score_fn(y[idx], d[idx], g0_fold, g1_fold, m_fold, th)
            n_k = psi_k.size
            s_k = float(psi_k.sum())
            ss_k = float((psi_k * psi_k).sum())
            total_n += n_k
            total_sum += s_k
            total_sumsq += ss_k
        if total_n > 1:
            mean_all = total_sum / total_n
            var_all = (total_sumsq - total_n * (mean_all ** 2)) / (total_n - 1)
            se_all = float(np.sqrt(max(var_all, 0.0) / total_n))
            tstat_strict = float(mean_all / se_all) if se_all > 0 else 0.0
        else:
            tstat_strict = 0.0
        pvalue_strict = float(2 * (1 - stats.norm.cdf(abs(tstat_strict))))

        # Choose which t-stat to report as main
        strict_applied = bool(strict_oos)
        main_tstat = float(tstat_strict if strict_applied else oos_tstat)
        main_pvalue = float(pvalue_strict if strict_applied else oos_pvalue)
    
    # === 2. ORTHOGONALITY DERIVATIVE TESTS ===
    # Create basis functions: constant + first few standardized confounders
    if data.confounders is None or len(data.confounders) == 0:
        raise ValueError("CausalData object must have confounders defined for orthogonality diagnostics")

    # If n_basis_funcs is not provided, default to (number of confounders + 1) for the constant term
    if n_basis_funcs is None:
        n_basis_funcs = len(data.confounders) + 1
    
    # Ensure confounders are represented as a float ndarray to avoid object-dtype reductions
    df_conf = data.get_df()[list(data.confounders)]
    X = df_conf.to_numpy(dtype=float)
    n_covs = min(max(n_basis_funcs - 1, 0), X.shape[1])  # -1 for constant term
    
    if n_covs > 0:
        X_selected = X[:, :n_covs]
        X_std = (X_selected - np.mean(X_selected, axis=0)) / (np.std(X_selected, axis=0) + 1e-8)
        X_basis = np.c_[np.ones(len(X)), X_std]  # Constant + standardized covariates
    else:
        X_basis = np.ones((len(X), 1))  # Only constant term
    
    if score_u == 'ATE':
        ortho_derivs_full = orthogonality_derivatives(X_basis, y, d, g0_outcomes, g1_outcomes, m_propensity, trimming_threshold=trimming_threshold)
        X_basis_trim = X_basis[trim_mask]
        ortho_derivs_trim = orthogonality_derivatives(X_basis_trim, y_trim, d_trim, g0_trim, g1_trim, m_trim, trimming_threshold=trimming_threshold)
        problematic_derivs_full = ortho_derivs_full[(np.abs(ortho_derivs_full['t_g1']) > 2) | (np.abs(ortho_derivs_full['t_g0']) > 2) | (np.abs(ortho_derivs_full['t_m']) > 2)]
        problematic_derivs_trim = ortho_derivs_trim[(np.abs(ortho_derivs_trim['t_g1']) > 2) | (np.abs(ortho_derivs_trim['t_g0']) > 2) | (np.abs(ortho_derivs_trim['t_m']) > 2)]
        derivs_interpretation = 'Large |t| (>2) indicate calibration issues'
        derivs_full_ok = len(problematic_derivs_full) == 0
        derivs_trim_ok = len(problematic_derivs_trim) == 0
    else:  # ATTE / ATT
        p_treated_full = float(np.mean(d))
        ortho_derivs_full = orthogonality_derivatives_atte(X_basis, y, d, g0_outcomes, m_propensity, p_treated_full, trimming_threshold=trimming_threshold)
        X_basis_trim = X_basis[trim_mask]
        p_treated_trim = float(np.mean(d_trim))
        ortho_derivs_trim = orthogonality_derivatives_atte(X_basis_trim, y_trim, d_trim, g0_trim, m_trim, p_treated_trim, trimming_threshold=trimming_threshold)
        # Backward-compatible alias columns for expected legacy names
        try:
            ortho_derivs_full = ortho_derivs_full.copy()
            ortho_derivs_full["d_g"] = ortho_derivs_full.get("d_g0", np.nan)
            ortho_derivs_full["t_g"] = ortho_derivs_full.get("t_g0", np.nan)
            ortho_derivs_full["d_m0"] = ortho_derivs_full.get("d_m", np.nan)
            ortho_derivs_full["t_m0"] = ortho_derivs_full.get("t_m", np.nan)
            ortho_derivs_trim = ortho_derivs_trim.copy()
            ortho_derivs_trim["d_g"] = ortho_derivs_trim.get("d_g0", np.nan)
            ortho_derivs_trim["t_g"] = ortho_derivs_trim.get("t_g0", np.nan)
            ortho_derivs_trim["d_m0"] = ortho_derivs_trim.get("d_m", np.nan)
            ortho_derivs_trim["t_m0"] = ortho_derivs_trim.get("t_m", np.nan)
        except Exception:
            pass
        problematic_derivs_full = ortho_derivs_full[(np.abs(ortho_derivs_full['t_g0']) > 2) | (np.abs(ortho_derivs_full['t_m']) > 2)]
        problematic_derivs_trim = ortho_derivs_trim[(np.abs(ortho_derivs_trim['t_g0']) > 2) | (np.abs(ortho_derivs_trim['t_m']) > 2)]
        derivs_interpretation = 'ATTE: check g0 & m only; large |t| (>2) => calibration issues'
        derivs_full_ok = len(problematic_derivs_full) == 0
        derivs_trim_ok = len(problematic_derivs_trim) == 0
    
    # === 3. INFLUENCE DIAGNOSTICS ===
    influence_full = influence_summary(y, d, g0_outcomes, g1_outcomes, m_propensity, theta_hat, score=score_u, trimming_threshold=trimming_threshold)

    # Re-estimate theta on the trimmed sample for fair trimmed influence diagnostics
    theta_hat_trim = theta_hat
    try:
        df_full_local = data.get_df()
        data_trim_obj = CausalData(
            df=df_full_local.loc[trim_mask].copy(),
            treatment=data.treatment.name,
            outcome=data.target.name,
            confounders=list(data.confounders) if data.confounders else None
        )
        res_trim = inference_fn(data_trim_obj, **inference_kwargs)
        theta_hat_trim = float(res_trim.get('coefficient', theta_hat))
    except Exception:
        pass

    influence_trim = influence_summary(y_trim, d_trim, g0_trim, g1_trim, m_trim, theta_hat_trim, score=score_u, trimming_threshold=trimming_threshold)
    
    # === DIAGNOSTIC ASSESSMENT ===
    model_se = result.get('std_error')
    if model_se is not None:
        se_reasonable = abs(influence_full['se_plugin'] - model_se) < 2 * model_se
    else:
        se_reasonable = False
    
    conditions = {
        'oos_moment_ok': abs(main_tstat) < 2.0,
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

    # ATTE-specific overlap and trim-sensitivity
    overlap_atte = None
    trim_curve_atte = None
    if score_u in ('ATTE', 'ATT'):
        overlap_atte = overlap_diagnostics_atte(m_propensity, d, eps_list=[0.95, 0.97, 0.98, 0.99])
        try:
            trim_curve_atte = trim_sensitivity_curve_atte(
                inference_fn, data, m_propensity, d,
                thresholds=np.linspace(0.90, 0.995, 12),
                **inference_kwargs
            )
        except Exception as _:
            trim_curve_atte = None
    
    # p-treated parameterization
    params_extra = {}
    if score_u in ('ATTE','ATT'):
        p_full = float(np.mean(d))
        p_trim = float(np.mean(d_trim)) if len(d_trim) > 0 else float('nan')
        params_extra = {
            'p_treated': p_full,
            'p_treated_full': p_full,
            'p_treated_trim': p_trim,
        }
    
    return {
        'theta': float(theta_hat),
        'params': {
            'score': score_u,
            'trimming_threshold': trimming_threshold,
            'strict_oos_requested': bool(strict_oos),
            'strict_oos_applied': bool(strict_applied),
            **params_extra,
        },
        'oos_moment_test': {
            'fold_results': oos_df,
            'tstat': float(main_tstat),
            'pvalue': float(main_pvalue),
            'tstat_fold_agg': float(oos_tstat),
            'pvalue_fold_agg': float(oos_pvalue),
            'tstat_strict': float(tstat_strict),
            'pvalue_strict': float(pvalue_strict),
            'aggregation': 'strict' if strict_applied else 'fold',
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
        'overlap_atte': overlap_atte,
        'robustness': {
            'trim_curve_atte': trim_curve_atte,
            'interpretation': 'ATTE typically more sensitive to trimming near m→1 (controls).'
        },
        'trimming_info': {
            'bounds': trim_propensity,
            'n_trimmed': int(n_trimmed),
            'pct_trimmed': float(n_trimmed / len(y) * 100.0)
        },
        'diagnostic_conditions': conditions,
        'overall_assessment': overall_assessment
    }


def orthogonality_derivatives_atte(
    X_basis: np.ndarray, y: np.ndarray, d: np.ndarray,
    g0: np.ndarray, m: np.ndarray, p_treated: float, trimming_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.

    For ψ_ATTE = [ D*(Y - g0 - θ)  -  (1-D)*(m/(1-m))*(Y - g0) ] / p_treated:

      ∂_{g0}[h] : (1/n) Σ h(X_i) * [ ((1-D_i)*m_i/(1-m_i) - D_i) / p_treated ]
      ∂_{m}[s]  : (1/n) Σ s(X_i) * [ -(1-D_i)*(Y_i - g0_i) / ( p_treated * (1-m_i)^2 ) ]

    Both have 0 expectation at the truth (Neyman orthogonality).
    """
    n, B = X_basis.shape
    m = np.clip(m, trimming_threshold, 1 - trimming_threshold)
    odds = m / (1.0 - m)

    dg0_terms = X_basis * (((1.0 - d) * odds - d) / (p_treated + 1e-12))[:, None]
    dg0 = dg0_terms.mean(axis=0)
    dg0_se = dg0_terms.std(axis=0, ddof=1) / np.sqrt(n)

    dm_terms = - X_basis * (((1.0 - d) * (y - g0)) / ((p_treated + 1e-12) * (1.0 - m)**2))[:, None]
    dm = dm_terms.mean(axis=0)
    dm_se = dm_terms.std(axis=0, ddof=1) / np.sqrt(n)

    dg1 = np.zeros(B)
    dg1_se = np.zeros(B)
    t = lambda est, se: est / np.maximum(se, 1e-12)

    return pd.DataFrame({
        "basis": np.arange(B),
        "d_g1": dg1, "se_g1": dg1_se, "t_g1": np.zeros(B),
        "d_g0": dg0, "se_g0": dg0_se, "t_g0": t(dg0, dg0_se),
        "d_m": dm,   "se_m": dm_se,   "t_m":  t(dm, dm_se),
    })




def overlap_diagnostics_atte(
    m: np.ndarray, d: np.ndarray, eps_list: List[float] = [0.95, 0.97, 0.98, 0.99]
) -> pd.DataFrame:
    """
    Key overlap metrics for ATTE: availability of suitable controls.
    Reports conditional shares: among CONTROLS, fraction with m(X) ≥ threshold; among TREATED, fraction with m(X) ≤ 1 - threshold.
    """
    rows = []
    m = np.asarray(m)
    d_bool = np.asarray(d).astype(bool)
    ctrl = ~d_bool
    trt = d_bool
    n_ctrl = int(ctrl.sum())
    n_trt = int(trt.sum())
    for thr in eps_list:
        pct_ctrl = float(100.0 * ((m[ctrl] >= thr).mean() if n_ctrl else np.nan))
        pct_trt = float(100.0 * ((m[trt] <= (1.0 - thr)).mean() if n_trt else np.nan))
        rows.append({
            "threshold": thr,
            "pct_controls_with_m_ge_thr": pct_ctrl,
            "pct_treated_with_m_le_1_minus_thr": pct_trt,
        })
    return pd.DataFrame(rows)




def trim_sensitivity_curve_atte(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    m: np.ndarray, d: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.90, 0.995, 12),
    **inference_kwargs
) -> pd.DataFrame:
    """
    Re-estimate θ while progressively trimming CONTROLS with large m(X).
    """
    df_full = data.get_df()
    rows = []
    for thr in thresholds:
        controls = (np.asarray(d).astype(bool) == False)
        high_p = (np.asarray(m) >= thr)
        drop = controls & high_p
        keep = ~drop
        df_trim = df_full.loc[keep].copy()
        data_trim = CausalData(
            df=df_trim, treatment=data.treatment.name,
            outcome=data.target.name, confounders=list(data.confounders) if data.confounders else None
        )
        res = inference_fn(data_trim, **inference_kwargs)
        rows.append({
            "trim_threshold": float(thr),
            "n": int(keep.sum()),
            "pct_dropped": float(100.0 * drop.mean()),
            "theta": float(res["coefficient"]),
            "se": float(res.get("std_error", np.nan))
        })
    return pd.DataFrame(rows)




def trim_sensitivity_curve_ate(
    m_hat: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    g0_hat: np.ndarray,
    g1_hat: np.ndarray,
    eps_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05),
) -> pd.DataFrame:
    """
    Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).

    For each epsilon in eps_grid, compute the AIPW/IRM ATE estimate using
    m_clipped = clip(m_hat, eps, 1-eps) over the full sample and report
    the plug-in standard error from the EIF.

    Parameters
    ----------
    m_hat, D, Y, g0_hat, g1_hat : np.ndarray
        Cross-fitted nuisances and observed arrays.
    eps_grid : tuple[float, ...]
        Sequence of clipping thresholds ε to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ['trim_eps','n','pct_clipped','theta','se'].
        pct_clipped is the percent of observations with m outside [ε,1-ε].
    """
    m = np.asarray(m_hat, dtype=float).ravel()
    d = np.asarray(D, dtype=float).ravel()
    y = np.asarray(Y, dtype=float).ravel()
    g0 = np.asarray(g0_hat, dtype=float).ravel()
    g1 = np.asarray(g1_hat, dtype=float).ravel()
    n = y.size
    rows: list[dict[str, float]] = []

    for eps in eps_grid:
        eps_f = float(eps)
        m_clip = np.clip(m, eps_f, 1.0 - eps_f)
        # share clipped (either side)
        pct_clipped = float(100.0 * np.mean((m <= eps_f) | (m >= 1.0 - eps_f))) if n > 0 else float("nan")
        # AIPW ATE with clipped propensity
        theta_terms = (g1 - g0) + d * (y - g1) / m_clip - (1.0 - d) * (y - g0) / (1.0 - m_clip)
        theta = float(np.mean(theta_terms)) if n > 0 else float("nan")
        psi = theta_terms - theta
        se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({
            "trim_eps": eps_f,
            "n": int(n),
            "pct_clipped": pct_clipped,
            "theta": theta,
            "se": se,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Placebo / robustness checks (moved from placebo.py)
# ------------------------------------------------------------------

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
    Provides a safe fallback for p_value if missing by using a normal approximation
    based on coefficient and std_error.
    """
    res = inference_fn(data, **kwargs)
    out = {"theta": float(res["coefficient"])}
    p = res.get("p_value")
    if p is None and "std_error" in res:
        try:
            se = float(res["std_error"])
            th = float(res["coefficient"])
            z = 0.0 if se == 0.0 else th / se
            p = 2.0 * (1 - stats.norm.cdf(abs(z)))
        except Exception:
            p = None
    out["p_value"] = float(p) if p is not None else float("nan")
    return out


# ------------------------------------------------------------------
# 1. Placebo ‑- generate random outcome
# ------------------------------------------------------------------

def _public_names(data: CausalData) -> tuple[str, str]:
    tname = getattr(getattr(data, "treatment", None), "name", None) or getattr(data, "_treatment", "D")
    yname = getattr(getattr(data, "target", None), "name", None) or getattr(data, "_target", "Y")
    return tname, yname


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
    tname, yname = _public_names(data)
    original_outcome = df_mod[yname]
    df_mod[yname] = _generate_random_outcome(original_outcome, rng)

    ck_mod = CausalData(
        df=df_mod,
        treatment=tname,
        outcome=yname,
        confounders=(list(data.confounders) if data.confounders else None),
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
    tname, yname = _public_names(data)
    original_treatment = df_mod[tname]
    df_mod[tname] = _generate_random_treatment(original_treatment, rng)

    ck_mod = CausalData(
        df=df_mod,
        treatment=tname,
        outcome=yname,
        confounders=(list(data.confounders) if data.confounders else None),
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

    tname, yname = _public_names(data)
    df_mod = df.iloc[idx].copy()
    ck_mod = CausalData(
        df=df_mod,
        treatment=tname,
        outcome=yname,
        confounders=(list(data.confounders) if data.confounders else None),
    )
    return _run_inference(inference_fn, ck_mod, **inference_kwargs)



def _fast_fold_thetas_from_psi(psi_a: np.ndarray, psi_b: np.ndarray, fold_indices: List[np.ndarray]) -> List[float]:
    """
    Compute leave-fold-out θ_{-k} using cached ψ_a, ψ_b.
    θ_{-k} = - mean_R(ψ_b) / mean_R(ψ_a), where R is the complement of fold k.
    """
    psi_a = np.asarray(psi_a, dtype=float).ravel()
    psi_b = np.asarray(psi_b, dtype=float).ravel()
    n = psi_a.size
    sum_a = float(psi_a.sum())
    sum_b = float(psi_b.sum())
    thetas: List[float] = []
    for idx in fold_indices:
        idx = np.asarray(idx, dtype=int)
        n_r = n - idx.size
        if n_r <= 0:
            thetas.append(0.0)
            continue
        sa = sum_a - float(psi_a[idx].sum())
        sb = sum_b - float(psi_b[idx].sum())
        mean_a = sa / n_r
        mean_b = sb / n_r
        denom = mean_a if abs(mean_a) > 1e-12 else -1.0  # robust guard against div-by-zero
        thetas.append(-mean_b / denom)
    return thetas


def oos_moment_check_from_psi(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    fold_indices: List[np.ndarray],
    *,
    strict: bool = False,
) -> Tuple[pd.DataFrame, float, Optional[float]]:
    """
    OOS moment check using cached ψ_a, ψ_b only.
    Returns (fold-wise DF, t_fold_agg, t_strict if requested).
    """
    psi_a = np.asarray(psi_a, dtype=float).ravel()
    psi_b = np.asarray(psi_b, dtype=float).ravel()
    fold_thetas = _fast_fold_thetas_from_psi(psi_a, psi_b, fold_indices)

    rows: List[Dict[str, Any]] = []
    total_n = 0
    total_sum = 0.0
    total_sumsq = 0.0

    for k, idx in enumerate(fold_indices):
        idx = np.asarray(idx, dtype=int)
        th = fold_thetas[k]
        # ψ_k = ψ_b[idx] + ψ_a[idx]*θ_{-k}
        psi_k = psi_b[idx] + psi_a[idx] * th
        n_k = psi_k.size
        m_k = float(psi_k.mean()) if n_k > 0 else 0.0
        v_k = float(psi_k.var(ddof=1)) if n_k > 1 else 0.0
        rows.append({"fold": k, "n": n_k, "psi_mean": m_k, "psi_var": v_k})

        if strict and n_k > 0:
            total_n += n_k
            s = float(psi_k.sum())
            total_sum += s
            total_sumsq += float((psi_k * psi_k).sum())

    df = pd.DataFrame(rows)

    # Fold-aggregated t-stat
    num = float((df["n"] * df["psi_mean"]).sum())
    den = float(np.sqrt((df["n"] * df["psi_var"]).sum()))
    t_fold = (num / den) if den > 0 else 0.0

    t_strict: Optional[float] = None
    if strict and total_n > 1:
        mean_all = total_sum / total_n
        var_all = (total_sumsq - total_n * (mean_all ** 2)) / (total_n - 1)
        se_all = float(np.sqrt(max(var_all, 0.0) / total_n))
        t_strict = (mean_all / se_all) if se_all > 0 else 0.0

    return df, float(t_fold), (None if t_strict is None else float(t_strict))


# ---- High-level entry point similar to overlap_validation.run_overlap_diagnostics ----
ResultLike = Dict[str, Any] | Any


def _extract_score_inputs_from_result(res: ResultLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Extract (y, d, g0, g1, m, theta, score, psi_a, psi_b, fold_indices) from a
    dml_ate/dml_att-like result dict or an IRM-like model instance.

    Supports two paths:
      - Dict with keys {'model', 'coefficient'} and optional 'diagnostic_data'.
      - IRM/DoubleML-like model object with .data, cross-fitted nuisances, and optionally psi caches.
    """
    model = None
    theta = float('nan')
    score_str = 'ATE'

    # 1) Result dict
    if isinstance(res, dict):
        if 'model' in res:
            model = res['model']
        if 'coefficient' in res:
            theta = float(res['coefficient'])
        # Try to infer score from result params if present
        params = res.get('params') or res.get('Parameters') or {}
        sc = params.get('score') if isinstance(params, dict) else None
        if sc is not None:
            score_str = str(sc).upper()
        # If diagnostic_data carries arrays, prefer them
        dd = res.get('diagnostic_data', {}) if isinstance(res.get('diagnostic_data', {}), dict) else {}
        m = dd['m_hat'] if 'm_hat' in dd else (dd['m'] if 'm' in dd else None)
        g0 = dd['g0_hat'] if 'g0_hat' in dd else (dd['g0'] if 'g0' in dd else None)
        g1 = dd['g1_hat'] if 'g1_hat' in dd else (dd['g1'] if 'g1' in dd else None)
        y = dd['y'] if 'y' in dd else (dd['Y'] if 'Y' in dd else None)
        d = dd['d'] if 'd' in dd else (dd['D'] if 'D' in dd else None)
        if all(v is not None for v in (y, d, g0, g1, m)):
            y = np.asarray(y, dtype=float).ravel()
            d = np.asarray(d, dtype=float).ravel()
            g0 = np.asarray(g0, dtype=float).ravel()
            g1 = np.asarray(g1, dtype=float).ravel()
            m = np.asarray(m, dtype=float).ravel()
            # Score fallback from coefficient if missing
            if np.isnan(theta) and 'coefficient' in res:
                theta = float(res['coefficient'])
            # psi caches from model if available
            if model is None:
                return y, d, g0, g1, m, theta, score_str, None, None, None
        # otherwise continue to model extraction below
    else:
        model = res

    if model is None:
        raise ValueError("Result must contain 'model' or provide diagnostic_data arrays.")

    # 2) Model extraction
    # Try to determine score setting
    sc_attr = getattr(model, 'score', None)
    if sc_attr is None:
        sc_attr = getattr(model, '_score', None)
    if sc_attr is not None:
        score_str = 'ATTE' if 'ATT' in str(sc_attr).upper() else 'ATE'

    # Cross-fitted nuisances
    m, g0, g1 = extract_nuisances(model)

    # Observed y,d from model.data
    df = None
    data_obj = getattr(model, 'data', None)
    if data_obj is not None and hasattr(data_obj, 'get_df'):
        df = data_obj.get_df()
        # Try robust attribute names
        if hasattr(data_obj, 'treatment') and hasattr(data_obj, 'target'):
            tname = data_obj.treatment.name
            yname = data_obj.target.name
        else:
            # Fallback to common attributes used in CausalData
            tname = getattr(data_obj, '_treatment', 'D')
            yname = getattr(data_obj, '_target', 'Y')
        d = df[tname].to_numpy(dtype=float)
        y = df[yname].to_numpy(dtype=float)
    else:
        raise ValueError("Model.data with get_df() is required to extract y and d.")

    # Theta from wrapper or model
    if np.isnan(theta):
        coef = getattr(model, 'coef_', None)
        if coef is not None:
            try:
                theta = float(coef[0])
            except Exception:
                theta = float(coef)
        else:
            theta = float('nan')

    # Optional fast-path caches
    psi_a = getattr(model, 'psi_a_', None)
    psi_b = getattr(model, 'psi_b_', None)
    folds = getattr(model, 'folds_', None)
    if psi_a is not None and psi_b is not None and folds is not None:
        # Build fold indices list
        folds_arr = np.asarray(folds, dtype=int)
        K = int(folds_arr.max() + 1) if folds_arr.size > 0 else 0
        fold_indices = [np.where(folds_arr == k)[0] for k in range(K)]
        return y, d, g0, g1, m, float(theta), score_str, np.asarray(psi_a, dtype=float), np.asarray(psi_b, dtype=float), fold_indices

    return y, d, g0, g1, m, float(theta), score_str, None, None, None



def run_score_diagnostics(
    res: ResultLike = None,
    *,
    y: Optional[np.ndarray] = None,
    d: Optional[np.ndarray] = None,
    g0: Optional[np.ndarray] = None,
    g1: Optional[np.ndarray] = None,
    m: Optional[np.ndarray] = None,
    theta: Optional[float] = None,
    score: Optional[str] = None,
    trimming_threshold: float = 0.01,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True,
) -> Dict[str, Any]:
    """
    Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

    You can call it in TWO ways:
      A) With raw arrays:
         run_score_diagnostics(y=..., d=..., g0=..., g1=..., m=..., theta=...)
      B) With a model/result:
         run_score_diagnostics(res=<dml_ate/dml_att result dict or IRM-like model>)

    Returns a dictionary with:
      - params (score, trimming_threshold)
      - oos_moment_test (if fast-path caches available on model; else omitted)
      - orthogonality_derivatives (DataFrame)
      - influence_diagnostics (full_sample)
      - summary (compact DataFrame) if return_summary=True
      - meta
    """
    # Resolve inputs
    psi_a = psi_b = None
    fold_indices = None
    if any(v is None for v in (y, d, g0, g1, m)):
        if res is None:
            raise ValueError("Pass either (y,d,g0,g1,m,theta) or `res`.")
        y, d, g0, g1, m, theta_ex, score_detected, psi_a, psi_b, fold_indices = _extract_score_inputs_from_result(res)
        if theta is None:
            theta = theta_ex
        if score is None:
            score = score_detected
    else:
        if score is None:
            score = 'ATE'

    score_u = str(score).upper()

    # Enforce binary D and scrub non-finite across inputs
    d = (np.asarray(d) > 0.5).astype(float)
    y_arr = np.asarray(y, dtype=float)
    g0_arr = np.asarray(g0, dtype=float)
    g1_arr = np.asarray(g1, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(d) & np.isfinite(g0_arr) & np.isfinite(g1_arr) & np.isfinite(m_arr)
    y, d, g0, g1, m = y_arr[mask], d[mask], g0_arr[mask], g1_arr[mask], m_arr[mask]

    # Influence diagnostics
    infl = influence_summary(y, d, g0, g1, m, float(theta if theta is not None else 0.0), score=score_u, trimming_threshold=trimming_threshold)

    # Build basis for orthogonality derivatives
    # If we can access confounders via res.model.data, use them; else constant-only basis
    X_basis = None
    use_data_basis = False
    if res is not None:
        model = res.get('model') if isinstance(res, dict) else res
        data_obj = getattr(model, 'data', None) if model is not None else None
        if data_obj is not None and hasattr(data_obj, 'get_df') and getattr(data_obj, 'confounders', None) is not None:
            df_conf = data_obj.get_df()[list(data_obj.confounders)]
            X = df_conf.to_numpy(dtype=float)
            if n_basis_funcs is None:
                n_basis_funcs = len(data_obj.confounders) + 1
            n_covs = min(max(n_basis_funcs - 1, 0), X.shape[1])
            if n_covs > 0:
                X_sel = X[:, :n_covs]
                X_std = (X_sel - np.mean(X_sel, axis=0)) / (np.std(X_sel, axis=0) + 1e-8)
                X_basis = np.c_[np.ones(X.shape[0]), X_std]
                use_data_basis = True
    if X_basis is None:
        X_basis = np.ones((len(y), 1))
    elif use_data_basis:
        # Align X_basis with scrubbed arrays
        try:
            X_basis = X_basis[mask]
        except Exception:
            # If masking fails due to shape issues, fall back to constant basis
            X_basis = np.ones((len(y), 1))

    if score_u == 'ATE':
        ortho = orthogonality_derivatives(X_basis, y, d, g0, g1, m, trimming_threshold=trimming_threshold)
    elif score_u in ('ATTE', 'ATT'):
        p_treated = float(np.mean(d))
        ortho = orthogonality_derivatives_atte(X_basis, y, d, g0, m, p_treated, trimming_threshold=trimming_threshold)
    else:
        raise ValueError("score must be 'ATE' or 'ATTE'")

    # Optional fast-path OOS moment check using cached psi if available
    oos = None
    if psi_a is not None and psi_b is not None and fold_indices is not None:
        df_oos, t_fold, t_strict = oos_moment_check_from_psi(psi_a, psi_b, fold_indices, strict=True)
        p_fold = float(2 * (1 - stats.norm.cdf(abs(t_fold))))
        p_strict = float(2 * (1 - stats.norm.cdf(abs(t_strict)))) if t_strict is not None else float('nan')
        oos = {
            'fold_results': df_oos,
            'tstat_fold_agg': float(t_fold),
            'pvalue_fold_agg': float(p_fold),
            'tstat_strict': (None if t_strict is None else float(t_strict)),
            'pvalue_strict': (float('nan') if t_strict is None else float(2 * (1 - stats.norm.cdf(abs(t_strict))))),
            'interpretation': 'Near 0 indicates moment condition holds.'
        }

    report: Dict[str, Any] = {
        'params': {
            'score': score_u,
            'trimming_threshold': float(trimming_threshold),
        },
        'orthogonality_derivatives': ortho,
        'influence_diagnostics': infl,
    }
    if oos is not None:
        report['oos_moment_test'] = oos

    if return_summary:
        # Build compact summary rows
        max_t_g1 = float(np.nanmax(np.abs(ortho['t_g1'])) if 't_g1' in ortho.columns else np.nan)
        max_t_g0 = float(np.nanmax(np.abs(ortho['t_g0'])) if 't_g0' in ortho.columns else np.nan)
        max_t_m  = float(np.nanmax(np.abs(ortho['t_m']))  if 't_m'  in ortho.columns else np.nan)
        rows = [
            {'metric': 'se_plugin', 'value': infl['se_plugin']},
            {'metric': 'psi_p99_over_med', 'value': infl['p99_over_med']},
            {'metric': 'psi_kurtosis', 'value': infl['kurtosis']},
            {'metric': 'max_|t|_g1', 'value': max_t_g1},
            {'metric': 'max_|t|_g0', 'value': max_t_g0},
            {'metric': 'max_|t|_m',  'value': max_t_m},
        ]
        if oos is not None:
            rows.append({'metric': 'oos_tstat_fold', 'value': oos['tstat_fold_agg']})
            rows.append({'metric': 'oos_tstat_strict', 'value': (np.nan if oos['tstat_strict'] is None else oos['tstat_strict'])})
        report['summary'] = pd.DataFrame(rows)

    report['meta'] = {'n': int(len(y)), 'score': score_u}
    # Augment with flags and thresholds, and enrich summary
    try:
        report = add_score_flags(report)
    except Exception:
        # be permissive: if flags fail, still return base report
        pass
    return report



# -------------- Flags and summary augmentation for score diagnostics --------------

def _grade(val: float, warn: float, strong: float, *, larger_is_worse: bool = True) -> str:
    """Map a scalar to GREEN/YELLOW/RED with optional direction."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "NA"
    v = float(val)
    if larger_is_worse:
        return "GREEN" if v < warn else ("YELLOW" if v < strong else "RED")
    else:
        # smaller is worse → invert thresholds
        return "GREEN" if v <= warn else ("YELLOW" if v <= strong else "RED")


def add_score_flags(rep_score: dict, thresholds: dict | None = None, *, effect_size_guard: float = 0.02, oos_gate: bool = True, se_rule: str | None = None, se_ref: float | None = None) -> dict:
    """
    Augment run_score_diagnostics(...) dict with:
      - rep['flags'] (per-metric flags)
      - rep['thresholds'] (the cutoffs used)
      - rep['summary'] with a new 'flag' column
      - rep['overall_flag'] (rollup)

    Additional logic:
      - Practical effect-size guard: if the constant-basis derivative magnitude is tiny
        (<= effect_size_guard), then downgrade an orthogonality RED to GREEN (if OOS is GREEN)
        or to YELLOW (otherwise). Controlled by `oos_gate`.
      - Huge-n relaxation: for very large n (>= 200k), relax tail/kurtosis flags slightly
        under specified value gates.
    """
    rep = deepcopy(rep_score)

    # --- defaults tuned to your style (>= means worse) ---
    thr = {
        # heavy tails / stability
        "tail_ratio_warn": 10.0,   # |psi| p99 / median
        "tail_ratio_strong": 20.0,
        "kurt_warn": 10.0,         # normal≈3; allow slack for EIFs
        "kurt_strong": 30.0,

        # orthogonality t-stats (|t| should be ~0)
        "t_warn": 2.0,
        "t_strong": 4.0,

        # OOS moment t-stat (|t| should be ~0)
        "oos_warn": 2.0,
        "oos_strong": 3.0,
    }
    if thresholds:
        thr.update(thresholds)

    # --- fetch raw pieces robustly ---
    infl = rep.get("influence_diagnostics", {}) or {}
    ortho = rep.get("orthogonality_derivatives")
    oos   = rep.get("oos_moment_test")

    # Influence metrics
    tail_ratio = infl.get("p99_over_med") if isinstance(infl, dict) else None
    kurtosis   = infl.get("kurtosis") if isinstance(infl, dict) else None

    # Max |t| over bases (compute if not already summarized)
    def _max_abs(df: pd.DataFrame, col: str) -> float | float:
        if isinstance(df, pd.DataFrame) and col in df.columns:
            a = np.abs(df[col].to_numpy(dtype=float))
            return float(np.nanmax(a)) if a.size else float("nan")
        return float("nan")

    max_t_g1 = _max_abs(ortho, "t_g1")
    max_t_g0 = _max_abs(ortho, "t_g0")
    max_t_m  = _max_abs(ortho, "t_m")

    # OOS t-stats (if available)
    oos_t_fold   = float(oos.get("tstat_fold_agg")) if isinstance(oos, dict) and "tstat_fold_agg" in oos else float("nan")
    oos_t_strict = float(oos.get("tstat_strict"))   if isinstance(oos, dict) and "tstat_strict"   in oos and oos.get("tstat_strict") is not None else float("nan")

    # --- compute flags ---
    flags = {
        # influence / tails
        "psi_tail_ratio": _grade(tail_ratio, thr["tail_ratio_warn"], thr["tail_ratio_strong"], larger_is_worse=True),
        "psi_kurtosis":   _grade(kurtosis,   thr["kurt_warn"],       thr["kurt_strong"],       larger_is_worse=True),

        # orthogonality derivatives
        "ortho_max_|t|_g1": _grade(max_t_g1, thr["t_warn"], thr["t_strong"], larger_is_worse=True),
        "ortho_max_|t|_g0": _grade(max_t_g0, thr["t_warn"], thr["t_strong"], larger_is_worse=True),
        "ortho_max_|t|_m":  _grade(max_t_m,  thr["t_warn"], thr["t_strong"], larger_is_worse=True),

        # OOS moment check (prefer strict if present)
        "oos_tstat_fold":   _grade(abs(oos_t_fold),   thr["oos_warn"], thr["oos_strong"], larger_is_worse=True),
        "oos_tstat_strict": _grade(abs(oos_t_strict), thr["oos_warn"], thr["oos_strong"], larger_is_worse=True),
    }

    # choose one OOS flag as canonical (strict if present)
    canonical_oos = "oos_tstat_strict" if flags["oos_tstat_strict"] != "NA" else "oos_tstat_fold"
    flags["oos_moment"] = flags[canonical_oos]

    # --- attach thresholds used ---
    rep["thresholds"] = {
        "tail_ratio_warn": thr["tail_ratio_warn"],
        "tail_ratio_strong": thr["tail_ratio_strong"],
        "kurt_warn": thr["kurt_warn"],
        "kurt_strong": thr["kurt_strong"],
        "t_warn": thr["t_warn"],
        "t_strong": thr["t_strong"],
        "oos_warn": thr["oos_warn"],
        "oos_strong": thr["oos_strong"],
    }

    # --- build/update compact summary ---
    # Start from existing summary if present; otherwise create it.
    summary_rows = [
        ("se_plugin", infl.get("se_plugin") if isinstance(infl, dict) else np.nan),
        ("psi_p99_over_med", tail_ratio),
        ("psi_kurtosis", kurtosis),
        ("max_|t|_g1", max_t_g1),
        ("max_|t|_g0", max_t_g0),
        ("max_|t|_m",  max_t_m),
    ]
    if not np.isnan(oos_t_fold):
        summary_rows.append(("oos_tstat_fold", oos_t_fold))
    if not np.isnan(oos_t_strict):
        summary_rows.append(("oos_tstat_strict", oos_t_strict))

    # Map summary metric → flag key and rule
    summary_flag_rules = {
        "psi_p99_over_med": ("psi_tail_ratio", True, thr["tail_ratio_warn"], thr["tail_ratio_strong"]),
        "psi_kurtosis":     ("psi_kurtosis",   True, thr["kurt_warn"],       thr["kurt_strong"]),
        "max_|t|_g1":       ("ortho_max_|t|_g1", True, thr["t_warn"], thr["t_strong"]),
        "max_|t|_g0":       ("ortho_max_|t|_g0", True, thr["t_warn"], thr["t_strong"]),
        "max_|t|_m":        ("ortho_max_|t|_m",  True, thr["t_warn"], thr["t_strong"]),
        "oos_tstat_fold":   ("oos_tstat_fold",   True, thr["oos_warn"], thr["oos_strong"]),
        "oos_tstat_strict": ("oos_tstat_strict", True, thr["oos_warn"], thr["oos_strong"]),
        # se_plugin is scale-dependent → mark NA by default
        "se_plugin":        (None, True, np.nan, np.nan),
    }

    df_sum = pd.DataFrame(summary_rows, columns=["metric", "value"])
    df_sum["flag"] = [
        (
            _grade(abs(v) if k.startswith("oos_tstat") else v,
                   summary_flag_rules[k][2], summary_flag_rules[k][3],
                   larger_is_worse=summary_flag_rules[k][1])
            if summary_flag_rules[k][0] is not None else "NA"
        )
        for k, v in zip(df_sum["metric"], df_sum["value"])
    ]

    # If run_score_diagnostics already produced a summary, align/merge on known metrics.
    if "summary" in rep and isinstance(rep["summary"], pd.DataFrame):
        # left-join by 'metric' and prefer new flags
        cur = rep["summary"].copy()
        cur = cur.drop(columns=[c for c in cur.columns if c not in ("metric","value","flag")], errors="ignore")
        rep["summary"] = (cur.drop(columns=["flag"], errors="ignore")
                            .merge(df_sum[["metric","flag"]], on="metric", how="left"))
    else:
        rep["summary"] = df_sum

    # Assign initial flags
    rep["flags"] = flags

    # --- SE flagging (optional) ---
    try:
        # thresholds defaults for SE relative gap
        thr.setdefault("se_rel_warn", 0.25)
        thr.setdefault("se_rel_strong", 0.50)
        se = float(infl.get("se_plugin", float("nan"))) if isinstance(infl, dict) else float("nan")
        se_flag = "NA"
        if se_rule is not None:
            rule = str(se_rule).lower()
            if rule == "model" and se_ref is not None and np.isfinite(se) and np.isfinite(float(se_ref)):
                rel = abs(se - float(se_ref)) / max(float(se_ref), 1e-12)
                se_flag = _grade(rel, thr["se_rel_warn"], thr["se_rel_strong"], larger_is_worse=True)
        # write back to flags and summary if available
        rep["flags"]["se_plugin"] = se_flag
        if "summary" in rep and isinstance(rep["summary"], pd.DataFrame):
            try:
                rep["summary"].loc[rep["summary"]["metric"] == "se_plugin", "flag"] = se_flag
            except Exception:
                pass
        # also expose thresholds used
        rep.setdefault("thresholds", {})
        rep["thresholds"]["se_rel_warn"] = thr["se_rel_warn"]
        rep["thresholds"]["se_rel_strong"] = thr["se_rel_strong"]
    except Exception:
        # keep permissive behavior if anything goes wrong
        pass

    # ================= Practical adjustments =================
    # 1) Effect-size guard for orthogonality REDs
    try:
        ortho_df = rep.get("orthogonality_derivatives")
        if isinstance(ortho_df, pd.DataFrame):
            if "basis" in ortho_df.columns and 0 in set(ortho_df["basis"].tolist()):
                row0 = ortho_df.loc[ortho_df["basis"] == 0]
                # Absolute derivative magnitudes at constant basis
                d_g1_abs = float(abs(row0["d_g1"].values[0])) if "d_g1" in ortho_df.columns else float("nan")
                d_g0_abs = float(abs(row0["d_g0"].values[0])) if "d_g0" in ortho_df.columns else float("nan")
                d_m_abs  = float(abs(row0["d_m" ].values[0])) if "d_m"  in ortho_df.columns else float("nan")
            else:
                d_g1_abs = d_g0_abs = d_m_abs = float("nan")
            oos_ok = (rep["flags"].get("oos_moment") == "GREEN") if oos_gate else True
            def soften(_):
                return "GREEN" if oos_ok else "YELLOW"
            if not np.isnan(d_g1_abs) and d_g1_abs <= effect_size_guard and rep["flags"].get("ortho_max_|t|_g1") == "RED":
                rep["flags"]["ortho_max_|t|_g1"] = soften(rep["flags"]["ortho_max_|t|_g1"])
            if not np.isnan(d_g0_abs) and d_g0_abs <= effect_size_guard and rep["flags"].get("ortho_max_|t|_g0") == "RED":
                rep["flags"]["ortho_max_|t|_g0"] = soften(rep["flags"]["ortho_max_|t|_g0"])
            if not np.isnan(d_m_abs) and d_m_abs <= effect_size_guard and rep["flags"].get("ortho_max_|t|_m") == "RED":
                rep["flags"]["ortho_max_|t|_m"] = soften(rep["flags"]["ortho_max_|t|_m"])
    except Exception:
        pass

    # 2) Huge-n synthetic run relaxation for tails
    try:
        n = int(rep.get("meta", {}).get("n", 0))
        if n >= 200_000:
            # Tail ratio: YELLOW→GREEN if value ≤ 12.0
            try:
                val_tail = float(df_sum.loc[df_sum["metric"] == "psi_p99_over_med", "value"].iloc[0])
            except Exception:
                val_tail = tail_ratio if tail_ratio is not None else float("nan")
            if rep["flags"].get("psi_tail_ratio") == "YELLOW" and np.isfinite(val_tail) and val_tail <= 12.0:
                rep["flags"]["psi_tail_ratio"] = "GREEN"
            # Kurtosis: RED→YELLOW if value ≤ 50.0
            try:
                val_kurt = float(df_sum.loc[df_sum["metric"] == "psi_kurtosis", "value"].iloc[0])
            except Exception:
                val_kurt = kurtosis if kurtosis is not None else float("nan")
            if rep["flags"].get("psi_kurtosis") == "RED" and np.isfinite(val_kurt) and val_kurt <= 50.0:
                rep["flags"]["psi_kurtosis"] = "YELLOW"
    except Exception:
        pass

    # 3) Recompute overall and sync summary flags with final per-metric flags
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2, "NA": -1}
    worst = max((order.get(f, -1) for f in rep["flags"].values()), default=-1)
    inv = {v: k for k, v in order.items()}
    rep["overall_flag"] = inv.get(worst, "NA")

    # Update summary flags in-place to reflect final flags
    map_flags = {
        "psi_p99_over_med": rep["flags"].get("psi_tail_ratio", "NA"),
        "psi_kurtosis":     rep["flags"].get("psi_kurtosis", "NA"),
        "max_|t|_g1":       rep["flags"].get("ortho_max_|t|_g1", "NA"),
        "max_|t|_g0":       rep["flags"].get("ortho_max_|t|_g0", "NA"),
        "max_|t|_m":        rep["flags"].get("ortho_max_|t|_m", "NA"),
        "oos_tstat_fold":   rep["flags"].get("oos_tstat_fold", "NA"),
        "oos_tstat_strict": rep["flags"].get("oos_tstat_strict", "NA"),
        "se_plugin":        rep["flags"].get("se_plugin", "NA"),
    }
    try:
        rep["summary"]["flag"] = rep["summary"]["metric"].map(map_flags).fillna(rep["summary"].get("flag"))
    except Exception:
        # If mapping fails, keep existing flags in summary
        pass

    return rep
