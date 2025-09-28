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

from causalkit.data.causaldata import CausalData
import warnings


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


def aipw_score_att(y: np.ndarray, d: np.ndarray, m0: np.ndarray, m1: np.ndarray, g: np.ndarray, theta: float, p1: Optional[float] = None, eps: float = 0.01) -> np.ndarray:
    """Deprecated: use aipw_score_atte(y,d,g0,g1,m,theta,p_treated,trimming_threshold)."""
    warnings.warn("aipw_score_att is deprecated; use aipw_score_atte with IRM naming.", DeprecationWarning, stacklevel=2)
    return aipw_score_atte(y, d, g0=m0, g1=m1, m=g, theta=theta, p_treated=p1, trimming_threshold=eps)


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
        Fitted internal IRM estimator (causalkit.inference.estimators.IRM) or a compatible dummy model
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
                     m: np.ndarray, theta_hat: float, k: int = 10, score: str = "ATE", trimming_threshold: float = 0.01, *, target: Optional[str] = None, clip_eps: Optional[float] = None) -> Dict[str, Any]:
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
    if target is not None:
        warnings.warn("'target' is deprecated; use 'score' ('ATE'|'ATTE').", DeprecationWarning, stacklevel=2)
        score = target
    if clip_eps is not None:
        warnings.warn("'clip_eps' is deprecated; use 'trimming_threshold'.", DeprecationWarning, stacklevel=2)
        trimming_threshold = clip_eps
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
    *,
    target: Optional[str] = None,
    clip_eps: float = 0.01,
    strict_oos: bool = False,
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
    >>> from causalkit.refutation.orthogonality import refute_irm_orthogonality
    >>> from causalkit.inference.ate.dml_ate import dml_ate
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

    # Handle deprecations and determine score
    if target is not None:
        warnings.warn("'target' is deprecated; use 'score' ('ATE'|'ATTE').", DeprecationWarning, stacklevel=2)
        if score is None:
            score = target
    if clip_eps is not None:
        warnings.warn("'clip_eps' is deprecated; use 'trimming_threshold'.", DeprecationWarning, stacklevel=2)
        trimming_threshold = clip_eps

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
    
    # Build backward-compatibility aliases for overlap diagnostics (ATT naming uses 'g')
    overlap_bc = overlap_atte.copy() if isinstance(overlap_atte, pd.DataFrame) else None
    if isinstance(overlap_bc, pd.DataFrame):
        if 'pct_controls_with_m_ge_thr' in overlap_bc.columns:
            overlap_bc['pct_controls_with_g_ge_thr'] = overlap_bc['pct_controls_with_m_ge_thr']
        if 'pct_treated_with_m_le_1_minus_thr' in overlap_bc.columns:
            overlap_bc['pct_treated_with_g_le_1_minus_thr'] = overlap_bc['pct_treated_with_m_le_1_minus_thr']
    
    # p-treated parameterization and legacy alias p1
    params_extra = {}
    if score_u in ('ATTE','ATT'):
        p_full = float(np.mean(d))
        p_trim = float(np.mean(d_trim)) if len(d_trim) > 0 else float('nan')
        params_extra = {
            'p_treated': p_full,
            'p_treated_full': p_full,
            'p_treated_trim': p_trim,
            # legacy aliases
            'p1': p_full,
            'p1_full': p_full,
            'p1_trim': p_trim,
        }
    
    # For ATT naming in derivative outputs, add legacy columns to copies (non-destructive)
    if score_u in ('ATTE','ATT') and isinstance(ortho_derivs_full, pd.DataFrame):
        def _with_legacy_cols(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()
            # map IRM naming to legacy ATT naming
            if 'd_g0' in df_out.columns:
                df_out['d_m0'] = df_out['d_g0']
                df_out['se_m0'] = df_out.get('se_g0', np.nan)
                df_out['t_m0'] = df_out.get('t_g0', np.nan)
            if 'd_m' in df_out.columns:
                df_out['d_g'] = df_out['d_m']
                df_out['se_g'] = df_out.get('se_m', np.nan)
                df_out['t_g'] = df_out.get('t_m', np.nan)
            # m1 is zero under ATTE
            df_out['d_m1'] = 0.0
            df_out['se_m1'] = 0.0
            df_out['t_m1'] = 0.0
            return df_out
        ortho_derivs_full = _with_legacy_cols(ortho_derivs_full)
        ortho_derivs_trim = _with_legacy_cols(ortho_derivs_trim)
    
    return {
        'theta': float(theta_hat),
        'params': {
            'score': score_u,
            'trimming_threshold': trimming_threshold,
            'strict_oos_requested': bool(strict_oos),
            'strict_oos_applied': bool(strict_applied),
            **params_extra,
            # backward-compat aliases
            'target': score_u,
            'clip_eps': trimming_threshold,
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
        'overlap_diagnostics': overlap_bc,
        'overlap_atte': overlap_atte,
        'robustness': {
            'trim_curve': trim_curve_atte,
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


def orthogonality_derivatives_att(
    X_basis: np.ndarray, y: np.ndarray, d: np.ndarray,
    m0: np.ndarray, g: np.ndarray, p1: float, eps: float = 0.01
) -> pd.DataFrame:
    """
    Deprecated ATT naming wrapper around orthogonality_derivatives_atte.
    Returns legacy column names for backward compatibility:
      - m1 derivatives are identically zero: d_m1,se_m1,t_m1
      - m0 maps to g0 in IRM naming
      - g maps to m in IRM naming (propensity)
    """
    warnings.warn(
        "orthogonality_derivatives_att is deprecated; use orthogonality_derivatives_atte with IRM naming.",
        DeprecationWarning,
        stacklevel=2,
    )
    df = orthogonality_derivatives_atte(X_basis, y, d, g0=m0, m=g, p_treated=p1, trimming_threshold=eps)
    # Build legacy columns alongside to preserve tests relying on old names
    out = pd.DataFrame({
        "basis": df["basis"].values,
        # legacy m1 (always zero under ATTE)
        "d_m1": np.zeros(len(df)),
        "se_m1": np.zeros(len(df)),
        "t_m1": np.zeros(len(df)),
        # legacy m0 corresponds to IRM g0
        "d_m0": df["d_g0"].values,
        "se_m0": df["se_g0"].values,
        "t_m0": df["t_g0"].values,
        # legacy g corresponds to IRM m (propensity)
        "d_g": df["d_m"].values,
        "se_g": df["se_m"].values,
        "t_g": df["t_m"].values,
    })
    return out


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


def overlap_diagnostics_att(
    g: np.ndarray, d: np.ndarray, eps_list: List[float] = [0.95, 0.97, 0.98, 0.99]
) -> pd.DataFrame:
    """Deprecated: use overlap_diagnostics_atte(m,d,eps_list)."""
    warnings.warn("overlap_diagnostics_att is deprecated; use overlap_diagnostics_atte with IRM naming.", DeprecationWarning, stacklevel=2)
    return overlap_diagnostics_atte(m=g, d=d, eps_list=eps_list)


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


def trim_sensitivity_curve_att(
    inference_fn: Callable[..., Dict[str, Any]],
    data: CausalData,
    g: np.ndarray, d: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.90, 0.995, 12),
    **inference_kwargs
) -> pd.DataFrame:
    """Deprecated: use trim_sensitivity_curve_atte(inference_fn,data,m,d,thresholds)."""
    warnings.warn("trim_sensitivity_curve_att is deprecated; use trim_sensitivity_curve_atte with IRM naming.", DeprecationWarning, stacklevel=2)
    return trim_sensitivity_curve_atte(inference_fn, data, m=g, d=d, thresholds=thresholds, **inference_kwargs)


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
    A valid causal rct_design should now yield θ ≈ 0 and a large p-value.
    """
    rng = np.random.default_rng(random_state)

    df_mod = data.get_df().copy()
    original_outcome = df_mod[data._target]
    df_mod[data._target] = _generate_random_outcome(original_outcome, rng)

    ck_mod = CausalData(
        df=df_mod,
        treatment=data._treatment,
        outcome=data._target,
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
    original_treatment = df_mod[data._treatment]
    df_mod[data._treatment] = _generate_random_treatment(original_treatment, rng)

    ck_mod = CausalData(
        df=df_mod,
        treatment=data._treatment,
        outcome=data._target,
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

    df_mod = df.iloc[idx].copy()
    ck_mod = CausalData(
        df=df_mod,
        treatment=data._treatment,
        outcome=data._target,
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
