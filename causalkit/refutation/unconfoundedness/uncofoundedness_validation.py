"""
Sensitivity analysis utilities for internal IRM-based estimators.

This module provides functions to produce a simple sensitivity-style report for
causal effect estimates returned by dml_ate and dml_att (which are based on the
internal IRM estimator). It no longer relies on DoubleML objects.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import warnings


def _compute_sensitivity_bias(sigma2: np.ndarray | float,
                               nu2: np.ndarray | float,
                               psi_sigma2: np.ndarray,
                               psi_nu2: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute (max) sensitivity bias and its IF, following DoubleML.
    max_bias = sqrt(max(sigma2 * nu2, 0)).
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    nu2_f = float(np.asarray(nu2).reshape(()))
    max_bias = float(np.sqrt(max(sigma2_f * nu2_f, 0.0)))
    denom = 2.0 * (max_bias if max_bias > 0 else 1.0)
    psi_max_bias = (sigma2_f * psi_nu2 + nu2_f * psi_sigma2) / denom
    return max_bias, psi_max_bias


def _combine_nu2(m_alpha: np.ndarray, rr: np.ndarray, cf_y: float, cf_d: float, rho: float) -> tuple[float, np.ndarray]:
    """Combine sensitivity levers into nu2 via per-unit quadratic form.

    nu2_i = (sqrt(2*m_alpha_i))^2 * cf_y + (|rr_i|)^2 * cf_d + 2*rho*sqrt(cf_y*cf_d)*|rr_i|*sqrt(2*m_alpha_i)
    Returns (nu2, psi_nu2) with psi_nu2 centered.
    """
    cf_y = float(cf_y)
    cf_d = float(cf_d)
    rho = float(np.clip(rho, -1.0, 1.0))
    if cf_y < 0 or cf_d < 0:
        raise ValueError("cf_y and cf_d must be >= 0.")
    a = np.sqrt(2.0 * np.maximum(np.asarray(m_alpha, dtype=float), 0.0))
    b = np.abs(np.asarray(rr, dtype=float))
    base = (a * a) * cf_y + (b * b) * cf_d + 2.0 * rho * np.sqrt(cf_y * cf_d) * a * b
    nu2 = float(np.mean(base))
    psi_nu2 = base - nu2
    return nu2, psi_nu2


def sensitivity_analysis(
    effect_estimation: Dict[str, Any],
    cf_y: float,
    cf_d: float,
    rho: float = 1.0,
    level: float = 0.95
) -> str:
    """
    Create a sensitivity-style summary for an IRM-based effect estimate.

    Parameters
    ----------
    effect_estimation : Dict[str, Any]
        A dictionary containing the effect estimation results from dml_ate or dml_att;
        must include:
        - 'model': A fitted IRM model object
        - 'coefficient', 'std_error', 'confidence_interval' are used if present
    cf_y : float
        Sensitivity parameter for the outcome equation (confounding strength)
    cf_d : float
        Sensitivity parameter for the treatment equation (confounding strength)
    rho : float, default 1.0
        Correlation parameter between unobserved confounders (for display only)
    level : float, default 0.95
        Confidence level for CI display

    Returns
    -------
    str
        A formatted sensitivity analysis summary report

    Notes
    -----
    This implementation prefers DoubleML-style model.sensitivity_analysis if available
    (for backward compatibility). Otherwise, it formats key statistics from the IRM
    model together with the provided sensitivity parameters for reporting.
    """
    # Validate inputs
    if not isinstance(effect_estimation, dict):
        raise TypeError("effect_estimation must be a dictionary")

    if 'model' not in effect_estimation:
        raise ValueError("effect_estimation must contain a 'model' key with a fitted IRM object")

    # Validate sensitivity inputs
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be in (0,1).")
    if cf_y < 0 or cf_d < 0:
        raise ValueError("cf_y and cf_d must be >= 0.")
    rho = float(np.clip(rho, -1.0, 1.0))

    model = effect_estimation['model']

    # Backward-compatibility: if model provides a sensitivity_analysis producing
    # model.sensitivity_summary, defer to it and return the provided summary.
    if hasattr(model, 'sensitivity_analysis'):
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
        if not hasattr(model, 'sensitivity_summary'):
            raise RuntimeError("Sensitivity analysis did not generate a summary")
        summary = model.sensitivity_summary
        effect_estimation['sensitivity_summary'] = summary
        return summary

    # IRM-based formatting path
    theta = effect_estimation.get('coefficient')
    se = effect_estimation.get('std_error')
    ci = effect_estimation.get('confidence_interval')

    # Fallback to model attributes if missing
    try:
        if theta is None and hasattr(model, 'coef'):
            theta = float(model.coef[0])
        if se is None and hasattr(model, 'se'):
            se = float(model.se[0])
        if ci is None and hasattr(model, 'confint'):
            ci_df = model.confint(level=level)
            if isinstance(ci_df, pd.DataFrame):
                ci = (float(ci_df.iloc[0, 0]), float(ci_df.iloc[0, 1]))
    except Exception:
        pass

    if theta is None or se is None or ci is None:
        # Preserve legacy behavior expected by some tests: when no DoubleML
        # sensitivity is available and not enough IRM stats are provided,
        # raise a TypeError similar to the previous implementation.
        raise TypeError("Insufficient stats for IRM fallback and no DoubleML support; provide coefficient, std_error, and confidence_interval or fit with IRM.")

    # Build a simple sensitivity-like summary table
    ci_lower, ci_upper = ci

    output_lines: List[str] = []
    output_lines.append("================== Sensitivity Analysis ==================")
    output_lines.append("")
    output_lines.append("------------------ Scenario          ------------------")
    output_lines.append(f"Significance Level: level={level}")
    output_lines.append(f"Sensitivity parameters: cf_y={cf_y}; cf_d={cf_d}, rho={rho}")
    output_lines.append("")
    output_lines.append("------------------ Bounds with CI    ------------------")
    header = f"{'':>6} {'CI lower':>11} {'theta lower':>12} {'theta':>15} {'theta upper':>12} {'CI upper':>13}"
    output_lines.append(header)

    # For a sensitivity-aware display, compute bias-based theta bounds using principled combination
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2.0)
    theta_hat = float(theta)
    try:
        if hasattr(model, "_sensitivity_element_est"):
            elems = model._sensitivity_element_est()
            m_alpha = elems.get("m_alpha")
            rr = elems.get("riesz_rep")
            if m_alpha is None or rr is None:
                warnings.warn("Model lacks m_alpha/riesz_rep; falling back to sampling CI for bounds.", RuntimeWarning)
                theta_lower = theta_hat - z * se
                theta_upper = theta_hat + z * se
            else:
                # Use local helpers to avoid fragile imports
                nu2_c, psi_nu2_c = _combine_nu2(m_alpha, rr, cf_y, cf_d, rho)
                max_bias, _ = _compute_sensitivity_bias(elems["sigma2"], nu2_c, elems["psi_sigma2"], psi_nu2_c)
                theta_lower = theta_hat - max_bias
                theta_upper = theta_hat + max_bias
        else:
            # Fallback: use sampling bound (does not reflect confounding)
            theta_lower = theta_hat - z * se
            theta_upper = theta_hat + z * se
    except Exception:
        theta_lower = theta_hat - z * se
        theta_upper = theta_hat + z * se

    row_name = getattr(getattr(model, 'data', None), 'treatment', None)
    if row_name is not None and hasattr(row_name, 'name'):
        row_name = row_name.name
    else:
        row_name = 'theta'

    output_lines.append(
        f"{row_name:>6} {ci_lower:11.6f} {theta_lower:12.6f} {theta_hat:15.6f} {theta_upper:12.6f} {ci_upper:13.6f}"
    )

    output_lines.append("")
    output_lines.append("------------------ Robustness (SNR proxy) -------------")
    rob_header = f"{'':>6} {'H_0':>6} {'SNR proxy (%)':>15} {'adj (%)':>8}"
    output_lines.append(rob_header)

    # Provide simple signal-to-noise proxy as a placeholder robustness value
    snr = abs(theta) / (se + 1e-12)
    rv = min(100.0, float(100.0 * (1.0 / (1.0 + snr))))
    rva = max(0.0, rv - 5.0)
    output_lines.append(f"{row_name:>6} {0.0:6.1f} {rv:15.6f} {rva:8.6f}")

    summary = "\n".join(output_lines)

    effect_estimation['sensitivity_summary'] = summary
    return summary


def validate_unconfoundedness_balance(
    effect_estimation: Dict[str, Any],
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Assess covariate balance under the unconfoundedness assumption by computing
    standardized mean differences (SMD) both before weighting (raw groups) and
    after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

    This function expects the result dictionary returned by dml_ate() or dml_att(),
    which includes a fitted IRM model and a 'diagnostic_data' entry with the
    necessary arrays.

    We compute, for each confounder X_j:
      - For ATE (weighted): w1 = D/m_hat, w0 = (1-D)/(1-m_hat).
      - For ATTE (weighted): Treated weight = 1 for D=1; Control weight w0 = m_hat/(1-m_hat) for D=0.
      - If estimation used normalized IPW (normalize_ipw=True), we scale the corresponding
        weights by their sample mean (as done in IRM) before computing balance.

    The SMD is defined as |mu1 - mu0| / s_pooled, where mu_g are (weighted) means in the
    (pseudo-)populations and s_pooled is the square root of the average of the (weighted)
    variances in the two groups.

    Parameters
    ----------
    effect_estimation : Dict[str, Any]
        Output dict from dml_ate() or dml_att(). Must contain 'model' and 'diagnostic_data'.
    threshold : float, default 0.1
        Threshold for SMD; values below indicate acceptable balance for most use cases.
    normalize : Optional[bool]
        Whether to use normalized weights. If None, inferred from effect_estimation['diagnostic_data']['normalize_ipw'].

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys:
        - 'smd': pd.Series of weighted SMD values indexed by confounder names
        - 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
        - 'score': 'ATE' or 'ATTE'
        - 'normalized': bool used for weighting
        - 'threshold': float
        - 'pass': bool indicating whether all weighted SMDs are below threshold
    """
    if not isinstance(effect_estimation, dict):
        raise TypeError("effect_estimation must be a dictionary produced by dml_ate() or dml_att()")
    if 'model' not in effect_estimation or 'diagnostic_data' not in effect_estimation:
        raise ValueError("Input must contain 'model' and 'diagnostic_data' (from dml_ate/dml_att)")

    diag = effect_estimation['diagnostic_data']
    if not isinstance(diag, dict):
        raise ValueError("'diagnostic_data' must be a dict")

    # Required arrays
    try:
        m_hat = np.asarray(diag['m_hat'], dtype=float)
        d = np.asarray(diag['d'], dtype=float)
        X = np.asarray(diag['x'], dtype=float)
        score = str(diag.get('score', '')).upper()
        used_norm = bool(diag.get('normalize_ipw', False)) if normalize is None else bool(normalize)
    except Exception as e:
        raise ValueError(f"diagnostic_data missing required fields: {e}")

    if score not in {"ATE", "ATTE"}:
        raise ValueError("diagnostic_data['score'] must be 'ATE' or 'ATTE'")
    if X.ndim != 2:
        raise ValueError("diagnostic_data['x'] must be a 2D array of shape (n, p)")
    n, p = X.shape
    if m_hat.shape[0] != n or d.shape[0] != n:
        raise ValueError("Length of m_hat and d must match number of rows in x")

    # Obtain confounder names if available
    try:
        model = effect_estimation['model']
        names = list(getattr(model.data, 'confounders', []))
        if not names or len(names) != p:
            names = [f"x{j+1}" for j in range(p)]
    except Exception:
        names = [f"x{j+1}" for j in range(p)]

    # Build weights per group according to estimand
    eps = 1e-12
    m = np.clip(m_hat, eps, 1.0 - eps)

    if score == "ATE":
        w1 = d / m
        w0 = (1.0 - d) / (1.0 - m)
        if used_norm:
            w1_mean = float(np.mean(w1))
            w0_mean = float(np.mean(w0))
            w1 = w1 / (w1_mean if w1_mean != 0 else 1.0)
            w0 = w0 / (w0_mean if w0_mean != 0 else 1.0)
    else:  # ATTE
        w1 = d  # treated weight = 1 for treated, 0 otherwise
        w0 = (1.0 - d) * (m / (1.0 - m))
        if used_norm:
            w1_mean = float(np.mean(w1))
            w0_mean = float(np.mean(w0))
            w1 = w1 / (w1_mean if w1_mean != 0 else 1.0)
            w0 = w0 / (w0_mean if w0_mean != 0 else 1.0)

    # Guard against no-mass groups
    s1 = float(np.sum(w1))
    s0 = float(np.sum(w0))
    if s1 <= 0 or s0 <= 0:
        raise RuntimeError("Degenerate weights: zero total mass in a pseudo-population")

    # Weighted means and variances (population-style)
    WX1 = (w1[:, None] * X)
    WX0 = (w0[:, None] * X)
    mu1 = WX1.sum(axis=0) / s1
    mu0 = WX0.sum(axis=0) / s0

    # Weighted variance: E[(X-mu)^2] under weight distribution
    var1 = (w1[:, None] * (X - mu1) ** 2).sum(axis=0) / s1
    var0 = (w0[:, None] * (X - mu0) ** 2).sum(axis=0) / s0
    sd1 = np.sqrt(np.maximum(var1, 0.0))
    sd0 = np.sqrt(np.maximum(var0, 0.0))
    s_pooled = np.sqrt(0.5 * (sd1 ** 2 + sd0 ** 2))

    # SMD with zero-guard (weighted)
    denom = np.where(s_pooled > 0, s_pooled, 1.0)
    smd = np.abs(mu1 - mu0) / denom
    smd_s = pd.Series(smd, index=names, dtype=float)

    # --- Unweighted (pre-weighting) SMD using raw treated/control groups ---
    mask1 = d.astype(bool)
    mask0 = ~mask1
    # If any group is empty (shouldn't be in typical settings), guard with nan SMDs
    if not np.any(mask1) or not np.any(mask0):
        smd_unw = np.full(p, np.nan, dtype=float)
    else:
        X1 = X[mask1]
        X0 = X[mask0]
        mu1_u = X1.mean(axis=0)
        mu0_u = X0.mean(axis=0)
        # population-style std to mirror weighted computation
        sd1_u = X1.std(axis=0, ddof=0)
        sd0_u = X0.std(axis=0, ddof=0)
        s_pool_u = np.sqrt(0.5 * (sd1_u ** 2 + sd0_u ** 2))
        denom_u = np.where(s_pool_u > 0, s_pool_u, 1.0)
        smd_unw = np.abs(mu1_u - mu0_u) / denom_u
    smd_unweighted_s = pd.Series(smd_unw, index=names, dtype=float)

    # Extra quick‑report fields
    smd_max = float(np.nanmax(smd)) if smd.size else float('nan')
    worst_features = smd_s.sort_values(ascending=False).head(10)

    out = {
        'smd': smd_s,
        'smd_unweighted': smd_unweighted_s,
        'score': score,
        'normalized': used_norm,
        'threshold': float(threshold),
        'pass': bool(np.all(np.isfinite(smd)) and np.nanmax(smd) < float(threshold)),
        'smd_max': smd_max,
        'worst_features': worst_features,
    }
    return out


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
    lower_lbl = f"{(1 - level) / 2 * 100:.1f} %"
    upper_lbl = f"{(0.5 + level / 2) * 100:.1f} %"
    for idx, row in summary.iterrows():
        # Format the row data - adjust column names based on actual DoubleML output
        row_name = str(idx) if not isinstance(idx, str) else idx
        try:
            ci_lower = row.get('ci_lower', row.get(lower_lbl, row.get('2.5 %', row.get('2.5%', 0.0))))
            theta_lower = row.get('theta_lower', row.get('theta lower', row.get('lower_bound', row.get('lower', 0.0))))
            theta = row.get('theta', row.get('estimate', row.get('coef', 0.0)))
            theta_upper = row.get('theta_upper', row.get('theta upper', row.get('upper_bound', row.get('upper', 0.0))))
            ci_upper = row.get('ci_upper', row.get(upper_lbl, row.get('97.5 %', row.get('97.5%', 0.0))))
            row_str = f"{row_name:>6} {ci_lower:11.6f} {theta_lower:12.6f} {theta:15.6f} {theta_upper:12.6f} {ci_upper:13.6f}"
            output_lines.append(row_str)
        except (KeyError, AttributeError):
            # Fallback formatting if exact column names differ
            row_values = [f"{val:11.6f}" if isinstance(val, (int, float)) else f"{val:>11}"
                          for val in list(row.values)[:5]]
            row_str = f"{row_name:>6} " + " ".join(row_values)
            output_lines.append(row_str)

    output_lines.append("")

    # Robustness SNR proxy section
    output_lines.append("------------------ Robustness (SNR proxy) -------------")

    # Create header for robustness values
    rob_header = f"{'':>6} {'H_0':>6} {'SNR proxy (%)':>15} {'adj (%)':>8}"
    output_lines.append(rob_header)

    # Add robustness values if present, else placeholders
    for idx, row in summary.iterrows():
        row_name = str(idx) if not isinstance(idx, str) else idx
        try:
            h_0 = row.get('H_0', row.get('null_hypothesis', 0.0))
            rv = row.get('RV', row.get('robustness_value', 0.0))
            rva = row.get('RVa', row.get('robustness_value_adjusted', 0.0))
            rob_row = f"{row_name:>6} {h_0:6.1f} {rv:15.6f} {rva:8.6f}"
            output_lines.append(rob_row)
        except (KeyError, AttributeError):
            rob_row = f"{row_name:>6} {0.0:6.1f} {0.0:15.6f} {0.0:8.6f}"
            output_lines.append(rob_row)

    return "\n".join(output_lines)


def get_sensitivity_summary(effect_estimation: Dict[str, Any]) -> Optional[str]:
    """
    Get the sensitivity summary string if available.

    Checks for a top-level 'sensitivity_summary' (set by sensitivity_analysis)
    and falls back to model.sensitivity_summary if present.
    """
    # Top-level string set by our function
    top = effect_estimation.get('sensitivity_summary') if isinstance(effect_estimation, dict) else None
    if isinstance(top, str):
        return top

    if not isinstance(effect_estimation, dict) or 'model' not in effect_estimation:
        return None

    model = effect_estimation['model']
    if hasattr(model, 'sensitivity_summary'):
        return model.sensitivity_summary
    return None


def sensitivity_benchmark(
    effect_estimation: Dict[str, Any],
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Computes a benchmark for a given set of features by refitting a short IRM model
    (excluding the provided features) and contrasting it with the original (long) model.
    Returns a DataFrame containing cf_y, cf_d, rho and the change in estimates.

    Parameters
    ----------
    effect_estimation : dict
        A dictionary containing the fitted IRM model under the key 'model'.
    benchmarking_set : list[str]
        List of confounder names to be used for benchmarking (to be removed in the short model).
    fit_args : dict, optional
        Additional keyword arguments for the IRM.fit() method of the short model.

    Returns
    -------
    pandas.DataFrame
        A one-row DataFrame indexed by the treatment name with columns:
        - cf_y, cf_d, rho: residual-based benchmarking strengths
        - theta_long, theta_short, delta: effect estimates and their change (long - short)
    """
    if not isinstance(effect_estimation, dict) or 'model' not in effect_estimation:
        raise TypeError("effect_estimation must be a dict containing a fitted IRM under key 'model'.")

    model = effect_estimation['model']

    # Validate model type by attribute presence (duck-typing IRM)
    required_attrs = ['data', 'coef_', 'se_', '_sensitivity_element_est']
    for attr in required_attrs:
        if not hasattr(model, attr):
            raise NotImplementedError("Sensitivity benchmarking requires a fitted IRM model with sensitivity elements.")

    # Extract current confounders
    try:
        x_list_long = list(getattr(model.data, 'confounders', []))
    except Exception as e:
        raise RuntimeError(f"Failed to access model data confounders: {e}")

    # input checks
    if not isinstance(benchmarking_set, list):
        raise TypeError(
            f"benchmarking_set must be a list. {str(benchmarking_set)} of type {type(benchmarking_set)} was passed."
        )
    if len(benchmarking_set) == 0:
        raise ValueError("benchmarking_set must not be empty.")
    if not set(benchmarking_set) <= set(x_list_long):
        raise ValueError(
            f"benchmarking_set must be a subset of features {str(x_list_long)}. "
            f"{str(benchmarking_set)} was passed."
        )
    if fit_args is not None and not isinstance(fit_args, dict):
        raise TypeError(f"fit_args must be a dict. {str(fit_args)} of type {type(fit_args)} was passed.")

    # Build short data excluding benchmarking features
    x_list_short = [x for x in x_list_long if x not in benchmarking_set]
    if len(x_list_short) == 0:
        raise ValueError("After removing benchmarking_set there are no confounders left to fit the short model.")

    # Create a shallow copy of the underlying DataFrame and build a new CausalData
    df_long = model.data.get_df()
    treatment_name = model.data.treatment.name
    target_name = model.data.target.name

    # Prefer in-scope names; fallback to import to avoid fragile self-import patterns
    try:
        CausalData  # type: ignore[name-defined]
        IRM  # type: ignore[name-defined]
    except NameError:
        from causalkit.data.causaldata import CausalData
        from causalkit.inference.estimators.irm import IRM

    data_short = CausalData(df=df_long, treatment=treatment_name, outcome=target_name, confounders=x_list_short)

    # Clone/construct a short IRM with same hyperparameters/learners
    irm_short = IRM(
        data=data_short,
        ml_g=model.ml_g,
        ml_m=model.ml_m,
        n_folds=getattr(model, 'n_folds', 4),
        n_rep=getattr(model, 'n_rep', 1),
        score=getattr(model, 'score', 'ATE'),
        normalize_ipw=getattr(model, 'normalize_ipw', False),
        trimming_rule=getattr(model, 'trimming_rule', 'truncate'),
        trimming_threshold=getattr(model, 'trimming_threshold', 1e-2),
        weights=getattr(model, 'weights', None),
        random_state=getattr(model, 'random_state', None),
    )

    # Fit short model
    if fit_args is None:
        irm_short.fit()
    else:
        irm_short.fit(**fit_args)

    # Long model stats
    theta_long = float(model.coef_[0])

    # Short model stats
    theta_short = float(irm_short.coef_[0])

    # Compute residual-based strengths on the long model
    df = model.data.get_df()
    y = df[target_name].to_numpy(dtype=float)
    d = df[treatment_name].to_numpy(dtype=float)
    m_hat = np.asarray(model.m_hat_, dtype=float)
    g0 = np.asarray(model.g0_hat_, dtype=float)
    g1 = np.asarray(model.g1_hat_, dtype=float)

    r_y = y - (d * g1 + (1.0 - d) * g0)
    r_d = d - m_hat

    def _center(a: np.ndarray) -> np.ndarray:
        return a - np.mean(a)

    def _ols_r2_and_fit(yv: np.ndarray, Z: np.ndarray) -> tuple[float, np.ndarray]:
        """Stable OLS on centered & standardized vars for R^2 and fitted component."""
        # Center the response
        yv_c = _center(yv)
        # Ensure 2D float array
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        # Center columns
        Zc = Z - np.nanmean(Z, axis=0, keepdims=True)
        # Standardize columns and drop near-constant / non-finite ones
        col_std = np.nanstd(Zc, axis=0, ddof=0)
        valid = np.isfinite(col_std) & (col_std > 1e-12)
        if not np.any(valid):
            # Nothing useful to project on
            yhat = np.zeros_like(yv_c)
            return 0.0, yhat
        Zcs = Zc[:, valid] / col_std[valid]
        # Replace any non-finite values to avoid numerical issues
        Zcs = np.nan_to_num(Zcs, nan=0.0, posinf=0.0, neginf=0.0)
        yv_c = np.nan_to_num(yv_c, nan=0.0, posinf=0.0, neginf=0.0)
        # Solve least squares with small rcond for stability
        from numpy.linalg import lstsq
        beta, *_ = lstsq(Zcs, yv_c, rcond=1e-12)
        yhat = Zcs @ beta
        # Guard R^2 computation
        denom = float(np.dot(yv_c, yv_c))
        if not np.isfinite(denom) or denom <= 1e-12:
            return 0.0, np.zeros_like(yv_c)
        num = float(np.dot(yhat, yhat))
        if not np.isfinite(num) or num < 0.0:
            return 0.0, np.zeros_like(yv_c)
        r2 = float(np.clip(num / denom, 0.0, 1.0))
        return r2, yhat

    Z = df[benchmarking_set].to_numpy(dtype=float)
    R2y, yhat_u = _ols_r2_and_fit(r_y, Z)
    R2d, dhat_u = _ols_r2_and_fit(r_d, Z)
    cf_y = float(R2y / max(1e-12, 1.0 - R2y))
    cf_d = float(R2d / max(1e-12, 1.0 - R2d))
    rho = float(np.corrcoef(_center(yhat_u), _center(dhat_u))[0, 1]) if (np.std(yhat_u) > 0 and np.std(dhat_u) > 0) else 0.0

    delta = theta_long - theta_short

    df_benchmark = pd.DataFrame(
        {
            "cf_y": [cf_y],
            "cf_d": [cf_d],
            "rho": [rho],
            "theta_long": [theta_long],
            "theta_short": [theta_short],
            "delta": [delta],
        },
        index=[treatment_name],
    )
    return df_benchmark