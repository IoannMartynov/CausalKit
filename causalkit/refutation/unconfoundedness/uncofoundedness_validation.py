"""
Sensitivity analysis utilities for internal IRM-based estimators.

This module provides functions to produce a simple sensitivity-style report for
causal effect estimates returned by dml_ate and dml_att (which are based on the
internal IRM estimator). It no longer relies on DoubleML objects.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings


def _compute_sensitivity_bias(sigma2: np.ndarray | float,
                               nu2: np.ndarray | float,
                               psi_sigma2: np.ndarray,
                               psi_nu2: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute (max) sensitivity bias and its IF, following DoubleML.
    max_bias = sqrt(max(sigma2 * nu2, 0)).

    Boundary behavior: if either sigma2 or nu2 is non-positive (<= 0),
    the bound is zero and the influence function should be identically zero.
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    nu2_f = float(np.asarray(nu2).reshape(()))
    # At the boundary, return zero bias and zero IF (theoretical behavior)
    if not (sigma2_f > 0.0 and nu2_f > 0.0):
        return 0.0, np.zeros_like(psi_sigma2, dtype=float)
    max_bias = float(np.sqrt(sigma2_f * nu2_f))
    denom = 2.0 * max_bias
    psi_max_bias = (sigma2_f * psi_nu2 + nu2_f * psi_sigma2) / denom
    return max_bias, psi_max_bias


def _combine_nu2(m_alpha: np.ndarray, rr: np.ndarray, cf_y: float, cf_d: float, rho: float) -> tuple[float, np.ndarray]:
    """Combine sensitivity levers into nu2 via per-unit quadratic form.

    nu2_i = (sqrt(2*m_alpha_i))^2 * cf_y + (|rr_i|)^2 * cf_d + 2*rho*sqrt(cf_y*cf_d)*|rr_i|*sqrt(2*m_alpha_i)
    Returns (nu2, psi_nu2) with psi_nu2 centered.

    Note: we use abs(rr) for a conservative worst-case cross-term; the quadratic
    form is PSD for signed rr as well, but abs() avoids reductions when rr < 0.
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

    # Fallback to model attributes if missing; support both coef_/coef and se_/se
    def _first_attr(obj, *names):
        for nm in names:
            if hasattr(obj, nm):
                val = getattr(obj, nm)
                try:
                    return float(val[0])
                except Exception:
                    try:
                        return float(val)
                    except Exception:
                        pass
        return None

    try:
        if theta is None:
            theta = _first_attr(model, 'coef_', 'coef')
        if se is None:
            se = _first_attr(model, 'se_', 'se')
        if ci is None and hasattr(model, 'confint'):
            try:
                ci_df = model.confint(level=level)
                if isinstance(ci_df, pd.DataFrame):
                    # Try named columns; fallback to first two
                    lower = None; upper = None
                    for col in ['ci_lower', f"{(1 - level) / 2 * 100:.1f} %", '2.5 %', '2.5%']:
                        if col in ci_df.columns:
                            lower = float(ci_df[col].iloc[0]); break
                    for col in ['ci_upper', f"{(0.5 + level / 2) * 100:.1f} %", '97.5 %', '97.5%']:
                        if col in ci_df.columns:
                            upper = float(ci_df[col].iloc[0]); break
                    if lower is None or upper is None:
                        lower = float(ci_df.iloc[0, 0])
                        upper = float(ci_df.iloc[0, 1])
                    ci = (lower, upper)
            except Exception:
                pass
        # CI fallback from theta and se
        if ci is None and (theta is not None and se is not None):
            from scipy.stats import norm
            z_ci = norm.ppf(0.5 + level / 2.0)
            ci = (float(theta) - z_ci * float(se), float(theta) + z_ci * float(se))
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
    output_lines.append("------------------ Robustness (risk proxy) -------------")
    rob_header = f"{'':>6} {'H_0':>6} {'risk proxy (%)':>15} {'adj (%)':>8}"
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
    # Pooled standard deviation directly from variances for clarity
    s_pooled = np.sqrt(0.5 * (np.maximum(var1, 0.0) + np.maximum(var0, 0.0)))

    # SMD with explicit zero-variance handling (weighted)
    smd = np.full(p, np.nan, dtype=float)
    zero_both = (var1 <= 1e-16) & (var0 <= 1e-16)
    diff = np.abs(mu1 - mu0)
    mask = (~zero_both) & (s_pooled > 1e-16)
    smd[mask] = diff[mask] / s_pooled[mask]
    smd[zero_both & (diff <= 1e-16)] = 0.0
    smd[zero_both & (diff > 1e-16)] = np.inf
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
        var1_u = sd1_u ** 2
        var0_u = sd0_u ** 2
        s_pool_u = np.sqrt(0.5 * (np.maximum(var1_u, 0.0) + np.maximum(var0_u, 0.0)))
        smd_unw = np.full(p, np.nan, dtype=float)
        zero_both_u = (var1_u <= 1e-16) & (var0_u <= 1e-16)
        diff_u = np.abs(mu1_u - mu0_u)
        mask_u = (~zero_both_u) & (s_pool_u > 1e-16)
        smd_unw[mask_u] = diff_u[mask_u] / s_pool_u[mask_u]
        smd_unw[zero_both_u & (diff_u <= 1e-16)] = 0.0
        smd_unw[zero_both_u & (diff_u > 1e-16)] = np.inf
    smd_unweighted_s = pd.Series(smd_unw, index=names, dtype=float)

    # Extra quick‑report fields
    smd_max = float(np.nanmax(smd)) if smd.size else float('nan')
    worst_features = smd_s.sort_values(ascending=False).head(10)

    # Decide pass/fail: ignore non-finite entries; also require low fraction of violations
    finite_mask = np.isfinite(smd_s.values)
    if np.any(finite_mask):
        frac_viol = float(np.mean(smd_s.values[finite_mask] >= float(threshold)))
        pass_bal = bool(np.all(smd_s.values[finite_mask] < float(threshold)) and (frac_viol < 0.10))
    else:
        # If all SMDs are non-finite (e.g., no variation across all features), treat as pass
        frac_viol = 0.0
        pass_bal = True

    out = {
        'smd': smd_s,
        'smd_unweighted': smd_unweighted_s,
        'score': score,
        'normalized': used_norm,
        'threshold': float(threshold),
        'pass': pass_bal,
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
    output_lines.append("------------------ Robustness (risk proxy) -------------")

    # Create header for robustness values
    rob_header = f"{'':>6} {'H_0':>6} {'risk proxy (%)':>15} {'adj (%)':>8}"
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

# ================= Unconfoundedness diagnostics (balance + overlap + weights) =================
from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Tuple as _Tuple, List as _List
from copy import deepcopy as _deepcopy  # not strictly needed, kept for future extensions


def _grade(val: float, warn: float, strong: float, *, larger_is_worse: bool = True) -> str:
    """Map a scalar to GREEN/YELLOW/RED; NA for nan/inf."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "NA"
    v = float(val)
    if larger_is_worse:
        # smaller values are better
        return "GREEN" if v < warn else ("YELLOW" if v < strong else "RED")
    else:
        # larger values are better
        return "GREEN" if v >= warn else ("YELLOW" if v >= strong else "RED")


def _safe_quantiles(a: np.ndarray, qs=(0.5, 0.9, 0.99)) -> _List[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(a, q)) for q in qs]


def _ks_unweighted(a: np.ndarray, b: np.ndarray) -> float:
    """Simple unweighted KS distance between two 1D arrays."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Evaluate ECDFs on merged grid
    grid = np.r_[a, b]
    grid.sort(kind="mergesort")
    Fa = np.searchsorted(a, grid, side="right") / a.size
    Fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))


def _extract_balance_inputs_from_result(
    res: _Dict[str, _Any] | _Any,
) -> _Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, _List[str]]:
    """
    Returns X (n,p), m_hat (n,), d (n,), score ('ATE'/'ATTE'), used_norm (bool), names (len p).
    Accepts:
      - dict with keys 'model' and 'diagnostic_data' (preferred)
      - model-like object with .data and cross-fitted m_hat_/predictions
    """
    # Result dict path
    if isinstance(res, dict):
        diag = res.get("diagnostic_data", {})
        if isinstance(diag, dict) and all(k in diag for k in ("x", "m_hat", "d")):
            X = np.asarray(diag["x"], dtype=float)
            m = np.asarray(diag["m_hat"], dtype=float).ravel()
            d = np.asarray(diag["d"], dtype=float).ravel()
            score = str(diag.get("score", "ATE")).upper()
            used_norm = bool(diag.get("normalize_ipw", False))
            names = diag.get("x_names")
            if not names or len(names) != X.shape[1]:
                # try model data for names
                model = res.get("model", None)
                if model is not None and getattr(model, "data", None) is not None:
                    try:
                        names = list(getattr(model.data, "confounders", []))
                    except Exception:
                        names = None
            if not names or len(names) != X.shape[1]:
                names = [f"x{j+1}" for j in range(X.shape[1])]
            return X, m, d, ("ATTE" if "ATT" in score else "ATE"), used_norm, names

        # Fall through to model extraction if diag missing
        res = res.get("model", res)

    # Model-like path
    model = res
    data_obj = getattr(model, "data", None)
    if data_obj is None or not hasattr(data_obj, "get_df"):
        raise ValueError("Could not extract data arrays. Provide `res` with diagnostic_data or a model with .data.get_df().")

    df = data_obj.get_df()
    confs = list(getattr(data_obj, "confounders", [])) or []
    if not confs:
        raise ValueError("CausalData must include confounders to compute balance (X).")
    X = df[confs].to_numpy(dtype=float)
    names = confs

    # m_hat
    if hasattr(model, "m_hat_") and getattr(model, "m_hat_", None) is not None:
        m = np.asarray(model.m_hat_, dtype=float).ravel()
    else:
        preds = getattr(model, "predictions", None)
        if isinstance(preds, dict) and "ml_m" in preds:
            m = np.asarray(preds["ml_m"], dtype=float).ravel()
        else:
            raise AttributeError("Could not locate propensity predictions (m_hat_ or predictions['ml_m']).")

    # d
    tname = getattr(getattr(data_obj, "treatment", None), "name", None) or getattr(data_obj, "_treatment", "D")
    d = df[tname].to_numpy(dtype=float).ravel()

    # score & normalization
    sc = getattr(model, "score", None) or getattr(model, "_score", "ATE")
    score = "ATTE" if "ATT" in str(sc).upper() else "ATE"
    used_norm = bool(getattr(model, "normalize_ipw", False))
    return X, m, d, score, used_norm, names


# ---------------- core SMD routine (adapted) ----------------

def _balance_smd(
    X: np.ndarray,
    d: np.ndarray,
    m_hat: np.ndarray,
    *,
    score: str,
    normalize: bool,
    threshold: float,
) -> _Dict[str, _Any]:
    """
    Compute weighted/unweighted SMDs + quick summaries using ATE/ATTE implied weights.
    """
    n, p = X.shape
    eps = 1e-12
    m = np.clip(np.asarray(m_hat, dtype=float).ravel(), eps, 1.0 - eps)
    d = np.asarray(d, dtype=float).ravel()
    score_u = str(score).upper()

    if score_u == "ATE":
        w1 = d / m
        w0 = (1.0 - d) / (1.0 - m)
    else:  # ATTE
        w1 = d
        w0 = (1.0 - d) * (m / (1.0 - m))

    if normalize:
        w1 /= float(np.mean(w1)) if np.mean(w1) != 0 else 1.0
        w0 /= float(np.mean(w0)) if np.mean(w0) != 0 else 1.0

    s1 = float(np.sum(w1))
    s0 = float(np.sum(w0))
    if s1 <= 0 or s0 <= 0:
        raise RuntimeError("Degenerate weights: zero total mass in a pseudo-population.")

    # weighted means/vars
    mu1 = (w1[:, None] * X).sum(axis=0) / s1
    mu0 = (w0[:, None] * X).sum(axis=0) / s0
    var1 = (w1[:, None] * (X - mu1) ** 2).sum(axis=0) / s1
    var0 = (w0[:, None] * (X - mu0) ** 2).sum(axis=0) / s0
    s_pool = np.sqrt(0.5 * (np.maximum(var1, 0.0) + np.maximum(var0, 0.0)))
    smd_w = np.full(p, np.nan, dtype=float)
    zero_both = (var1 <= 1e-16) & (var0 <= 1e-16)
    diff = np.abs(mu1 - mu0)
    mask = (~zero_both) & (s_pool > 1e-16)
    smd_w[mask] = diff[mask] / s_pool[mask]
    smd_w[zero_both & (diff <= 1e-16)] = 0.0
    smd_w[zero_both & (diff > 1e-16)] = np.inf

    # raw SMD
    mask1 = d.astype(bool)
    mask0 = ~mask1
    if not np.any(mask1) or not np.any(mask0):
        smd_u = np.full(p, np.nan)
    else:
        mu1_u = X[mask1].mean(axis=0)
        mu0_u = X[mask0].mean(axis=0)
        sd1_u = X[mask1].std(axis=0, ddof=0)
        sd0_u = X[mask0].std(axis=0, ddof=0)
        var1_u = sd1_u ** 2
        var0_u = sd0_u ** 2
        s_pool_u = np.sqrt(0.5 * (np.maximum(var1_u, 0.0) + np.maximum(var0_u, 0.0)))
        smd_u = np.full(p, np.nan)
        zero_both_u = (var1_u <= 1e-16) & (var0_u <= 1e-16)
        diff_u = np.abs(mu1_u - mu0_u)
        mask_u = (~zero_both_u) & (s_pool_u > 1e-16)
        smd_u[mask_u] = diff_u[mask_u] / s_pool_u[mask_u]
        smd_u[zero_both_u & (diff_u <= 1e-16)] = 0.0
        smd_u[zero_both_u & (diff_u > 1e-16)] = np.inf

    # summaries
    finite = np.isfinite(smd_w)
    smd_max = float(np.nanmax(smd_w)) if np.any(finite) else float("nan")
    frac_viol = float(np.mean(smd_w[finite] >= float(threshold))) if np.any(finite) else 0.0

    return {
        "smd_weighted": smd_w,
        "smd_unweighted": smd_u,
        "smd_max": smd_max,
        "frac_violations": frac_viol,
        "weights": (w1, w0),
        "mass": (s1, s0),
    }


# ---------------- main entry point ----------------

def run_unconfoundedness_diagnostics(
    *,
    res: _Dict[str, _Any] | _Any = None,
    X: _Optional[np.ndarray] = None,
    d: _Optional[np.ndarray] = None,
    m_hat: _Optional[np.ndarray] = None,
    names: _Optional[_List[str]] = None,
    score: _Optional[str] = None,
    normalize: _Optional[bool] = None,
    threshold: float = 0.10,
    eps_overlap: float = 0.01,
    return_summary: bool = True,
) -> _Dict[str, _Any]:
    """
    Single entry-point for *unconfoundedness* validation:
      - Covariate balance (SMD) after implied IPW/ATT weighting
      - Overlap diagnostics (propensity distribution by group)
      - Weight stability (ESS, tails, concentration)

    Usage:
      A) With a result dict / model:
         run_unconfoundedness_diagnostics(res=ate_result, threshold=0.10)
      B) With raw arrays:
         run_unconfoundedness_diagnostics(X=X, d=d, m_hat=m, score="ATE", normalize=False)

    Returns a dict with:
      - params (score, threshold, eps_overlap, normalize)
      - balance (SMD series, unweighted SMD, pass, worst_features)
      - weights (ESS, quantiles, top1 mass)
      - overlap (KS distance, tail shares, group min/max m)
      - summary (metric/value/flag)
      - thresholds, flags, overall_flag
    """
    # ---- Resolve inputs ----
    if (X is None or d is None or m_hat is None) and res is None:
        raise ValueError("Pass either (X, d, m_hat) or `res` with diagnostic_data/model.")

    if X is None or d is None or m_hat is None:
        X, m_hat, d, score_auto, used_norm_auto, names_auto = _extract_balance_inputs_from_result(res)
        if score is None:
            score = score_auto
        if normalize is None:
            normalize = used_norm_auto
        if names is None:
            names = names_auto
    else:
        if score is None:
            score = "ATE"
        if normalize is None:
            normalize = False
        if names is None:
            names = [f"x{j+1}" for j in range(X.shape[1])]

    score_u = str(score).upper()
    used_norm = bool(normalize)
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    m_hat = np.asarray(m_hat, dtype=float).ravel()
    n, p = X.shape
    if m_hat.size != n or d.size != n:
        raise ValueError("X, d, m_hat must have matching length n.")

    # ---- Balance (SMD) ----
    bal = _balance_smd(X, d, m_hat, score=score_u, normalize=used_norm, threshold=threshold)
    w1, w0 = bal["weights"]
    s1, s0 = bal["mass"]

    smd_w = pd.Series(bal["smd_weighted"], index=names, dtype=float, name="SMD_weighted")
    smd_u = pd.Series(bal["smd_unweighted"], index=names, dtype=float, name="SMD_unweighted")
    worst = smd_w.sort_values(ascending=False).head(10)

    # ---- Weight stability ----
    def _ess(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        s = float(np.sum(w))
        ss = float(np.sum(w * w)) + 1e-12
        return (s * s) / ss

    ess1 = _ess(w1); ess0 = _ess(w0)
    # group counts (unweighted)
    n1 = int((d > 0.5).sum())
    n0 = int((d <= 0.5).sum())
    # Clamp ratios to [0,1] to avoid numerical noise
    ess1_ratio = float(min(1.0, max(0.0, ess1 / max(n1, 1))))
    ess0_ratio = float(min(1.0, max(0.0, ess0 / max(n0, 1))))

    # weight tails & concentration (per group)
    q1 = _safe_quantiles(w1); q0 = _safe_quantiles(w0)
    tail_ratio1 = (q1[-1] / (q1[0] + 1e-12)) if np.isfinite(q1[-1]) and np.isfinite(q1[0]) else float("nan")
    tail_ratio0 = (q0[-1] / (q0[0] + 1e-12)) if np.isfinite(q0[-1]) and np.isfinite(q0[0]) else float("nan")
    # top 1% mass share
    def _top1_share(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        if w.size == 0:
            return float("nan")
        s = float(np.sum(w))
        k = max(1, int(np.ceil(0.01 * w.size)))
        return float(np.sum(np.sort(w)[-k:]) / (s + 1e-12))
    top1_1 = _top1_share(w1); top1_0 = _top1_share(w0)

    # ---- Overlap diagnostics on m_hat ----
    m = np.asarray(m_hat, dtype=float)
    m1 = m[d > 0.5]; m0 = m[d <= 0.5]
    ks_m = _ks_unweighted(m1, m0)
    pct_low = float(100.0 * np.mean(m <= eps_overlap))
    pct_high = float(100.0 * np.mean(m >= 1.0 - eps_overlap))
    pct_low_by_group = float(100.0 * np.mean(m1 <= eps_overlap)) if m1.size else float("nan")
    pct_high_by_group = float(100.0 * np.mean(m0 >= 1.0 - eps_overlap)) if m0.size else float("nan")
    m_stats = {
        "treated": dict(n=n1, m_min=float(np.min(m1)) if m1.size else float("nan"), m_max=float(np.max(m1)) if m1.size else float("nan")),
        "control": dict(n=n0, m_min=float(np.min(m0)) if m0.size else float("nan"), m_max=float(np.max(m0)) if m0.size else float("nan")),
    }

    # ---- Compose report ----
    report: _Dict[str, _Any] = {
        "params": {
            "score": score_u,
            "threshold": float(threshold),
            "eps_overlap": float(eps_overlap),
            "normalize": used_norm,
        },
        "balance": {
            "smd": smd_w,
            "smd_unweighted": smd_u,
            "smd_max": float(bal["smd_max"]),
            "frac_violations": float(bal["frac_violations"]),
            "pass": (bool(np.all(smd_w[np.isfinite(smd_w)] < float(threshold)) and float(bal["frac_violations"]) < 0.10) if np.any(np.isfinite(smd_w)) else True),
            "worst_features": worst,
        },
        "weights": {
            "sum_treated": s1,
            "sum_control": s0,
            "ess_treated": float(ess1),
            "ess_control": float(ess0),
            "ess_treated_ratio": float(ess1_ratio),
            "ess_control_ratio": float(ess0_ratio),
            "w_treated_quantiles": {"p50": q1[0], "p90": q1[1], "p99": q1[2]},
            "w_control_quantiles": {"p50": q0[0], "p90": q0[1], "p99": q0[2]},
            "w_tail_ratio_treated": float(tail_ratio1),
            "w_tail_ratio_control": float(tail_ratio0),
            "top1_mass_share_treated": float(top1_1),
            "top1_mass_share_control": float(top1_0),
        },
        "overlap": {
            "ks_m_treated_vs_control": float(ks_m),
            "pct_m_le_eps": pct_low,
            "pct_m_ge_1_minus_eps": pct_high,
            "pct_treated_m_le_eps": pct_low_by_group,
            "pct_control_m_ge_1_minus_eps": pct_high_by_group,
            "pct_m_outside_both": float(min(100.0, pct_low + pct_high)),
            "by_group_min_max": m_stats,
        },
        "meta": {"n": int(n), "p": int(p)},
    }

    # ---- Thresholds & flags (GREEN/YELLOW/RED) ----
    thr = {
        "smd_warn": 0.10, "smd_strong": 0.20,                   # balance
        "viol_frac_warn": 0.10, "viol_frac_strong": 0.25,       # share of features over threshold
        "ess_ratio_warn": 0.80, "ess_ratio_strong": 0.60,       # ESS/N_g (larger is better)
        "w_tail_warn": 20.0, "w_tail_strong": 50.0,             # p99/median of weights
        "top1_share_warn": 0.50, "top1_share_strong": 0.70,     # mass in top 1%
        "ks_warn": 0.25, "ks_strong": 0.40,                     # KS distance on m
        "extreme_m_warn": 10.0, "extreme_m_strong": 20.0,       # % overall outside [eps, 1-eps]
    }

    flags = {
        "balance_max_smd": _grade(report["balance"]["smd_max"], thr["smd_warn"], thr["smd_strong"], larger_is_worse=True),
        "balance_violations": _grade(report["balance"]["frac_violations"], thr["viol_frac_warn"], thr["viol_frac_strong"], larger_is_worse=True),
        "ess_treated_ratio": _grade(report["weights"]["ess_treated_ratio"], thr["ess_ratio_warn"], thr["ess_ratio_strong"], larger_is_worse=False),
        "ess_control_ratio": _grade(report["weights"]["ess_control_ratio"], thr["ess_ratio_warn"], thr["ess_ratio_strong"], larger_is_worse=False),
        "w_tail_treated": _grade(report["weights"]["w_tail_ratio_treated"], thr["w_tail_warn"], thr["w_tail_strong"], larger_is_worse=True),
        "w_tail_control": _grade(report["weights"]["w_tail_ratio_control"], thr["w_tail_warn"], thr["w_tail_strong"], larger_is_worse=True),
        "top1_mass_treated": _grade(report["weights"]["top1_mass_share_treated"], thr["top1_share_warn"], thr["top1_share_strong"], larger_is_worse=True),
        "top1_mass_control": _grade(report["weights"]["top1_mass_share_control"], thr["top1_share_warn"], thr["top1_share_strong"], larger_is_worse=True),
        "ks_propensity": _grade(report["overlap"]["ks_m_treated_vs_control"], thr["ks_warn"], thr["ks_strong"], larger_is_worse=True),
        "extreme_propensity_overall": _grade(
            max(report["overlap"]["pct_m_le_eps"], report["overlap"]["pct_m_ge_1_minus_eps"]),
            thr["extreme_m_warn"], thr["extreme_m_strong"], larger_is_worse=True
        ),
    }

    report["thresholds"] = thr
    report["flags"] = flags

    # Overall = worst of flags (NA ignored)
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2, "NA": -1}
    worst_flag = max((order.get(f, -1) for f in flags.values()), default=-1)
    inv = {v: k for k, v in order.items()}
    report["overall_flag"] = inv.get(worst_flag, "NA")

    # ---- Compact summary table ----
    if return_summary:
        rows = [
            ("balance_max_smd", report["balance"]["smd_max"], flags["balance_max_smd"]),
            ("balance_frac_violations", report["balance"]["frac_violations"], flags["balance_violations"]),
            ("ess_treated_ratio", report["weights"]["ess_treated_ratio"], flags["ess_treated_ratio"]),
            ("ess_control_ratio", report["weights"]["ess_control_ratio"], flags["ess_control_ratio"]),
            ("w_tail_ratio_treated", report["weights"]["w_tail_ratio_treated"], flags["w_tail_treated"]),
            ("w_tail_ratio_control", report["weights"]["w_tail_ratio_control"], flags["w_tail_control"]),
            ("top1_mass_share_treated", report["weights"]["top1_mass_share_treated"], flags["top1_mass_treated"]),
            ("top1_mass_share_control", report["weights"]["top1_mass_share_control"], flags["top1_mass_control"]),
            ("ks_m_treated_vs_control", report["overlap"]["ks_m_treated_vs_control"], flags["ks_propensity"]),
            ("pct_m_outside_overlap", max(report["overlap"]["pct_m_le_eps"], report["overlap"]["pct_m_ge_1_minus_eps"]), flags["extreme_propensity_overall"]),
        ]
        report["summary"] = pd.DataFrame(rows, columns=["metric", "value", "flag"])

    return report
