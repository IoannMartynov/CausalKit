"""
Sensitivity functions refactored into a dedicated module.

This module centralizes bias-aware sensitivity helpers and the public
entry points used by refutation utilities for unconfoundedness.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd

__all__ = ["sensitivity_analysis", "sensitivity_benchmark", "get_sensitivity_summary"]

# ---------------- Internals ----------------

_ESSENTIALLY_ZERO = 1e-32


# ---------------- Core sensitivity primitives (public, legacy-compatible) ----------------

def _compute_sensitivity_bias_unified(
    sigma2: np.ndarray | float,
    nu2: np.ndarray | float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    max_bias = sqrt(max(sigma2 * nu2, 0)). Influence function via delta method.
    Returns zero IF on the boundary and an IF shaped like psi_sigma2 otherwise.
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    nu2_f = float(np.asarray(nu2).reshape(()))
    if not (sigma2_f > 0.0 and nu2_f > 0.0):
        return 0.0, np.zeros_like(psi_sigma2, dtype=float)
    max_bias = float(np.sqrt(sigma2_f * nu2_f))
    denom = 2.0 * max_bias if max_bias > _ESSENTIALLY_ZERO else 1.0
    psi_sigma2 = np.asarray(psi_sigma2, float)
    psi_sigma2 = psi_sigma2 - float(np.mean(psi_sigma2))
    psi_nu2 = np.asarray(psi_nu2, float)
    psi_nu2 = psi_nu2 - float(np.mean(psi_nu2))
    psi_max_bias = (sigma2_f * psi_nu2 + nu2_f * psi_sigma2) / denom
    return max_bias, psi_max_bias

# Backward-compatible alias
def _compute_sensitivity_bias(
    sigma2: np.ndarray | float,
    nu2: np.ndarray | float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


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
    # numeric PSD clamp
    base = np.maximum(base, 0.0)
    nu2 = float(np.mean(base))
    psi_nu2 = base - nu2
    return nu2, psi_nu2


# ---------------- Bias-aware helpers (local variants + pullers) ----------------

def _combine_nu2_local(m_alpha: np.ndarray, rr: np.ndarray, cf_y: float, cf_d: float, rho: float, *, use_signed_rr: bool) -> tuple[float, np.ndarray]:
    """Nu^2 via per-unit quadratic form with optional sign-preserving rr."""
    cf_y = float(cf_y); cf_d = float(cf_d); rho = float(np.clip(rho, -1.0, 1.0))
    if cf_y < 0 or cf_d < 0:
        raise ValueError("cf_y and cf_d must be >= 0.")
    a = np.sqrt(2.0 * np.maximum(np.asarray(m_alpha, float), 0.0))
    b = np.asarray(rr, float)
    if not use_signed_rr:
        b = np.abs(b)  # worst-case sign
    base = (a * a) * cf_y + (b * b) * cf_d + 2.0 * rho * np.sqrt(cf_y * cf_d) * a * b
    base = np.maximum(base, 0.0)
    nu2 = float(np.mean(base))
    psi_nu2 = base - nu2
    return nu2, psi_nu2




def _compute_sensitivity_bias_local(
    sigma2: float,
    nu2: float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Backward-compatible wrapper delegating to unified helper."""
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


def _pull_theta_se_ci(effect_estimation: Dict[str, Any], level: float) -> tuple[float, float, tuple[float, float]]:
    """Robustly extract θ, se, and sampling CI."""
    from scipy.stats import norm as _norm
    model = effect_estimation['model']
    # theta
    try:
        theta = float(effect_estimation.get('coefficient', float(model.coef_[0])))
    except Exception:
        theta = float(model.coef_[0])
    # se
    try:
        se = float(effect_estimation.get('std_error', float(model.se_[0])))
    except Exception:
        se = float(model.se_[0])
    # sampling CI
    ci = effect_estimation.get('confidence_interval', None)
    if ci is None and hasattr(model, 'confint'):
        try:
            ci_df = model.confint(level=level)
            if isinstance(ci_df, pd.DataFrame):
                lower = None; upper = None
                for col in ['ci_lower', f"{(1-level)/2*100:.1f} %", '2.5 %', '2.5%']:
                    if col in ci_df.columns:
                        lower = float(ci_df[col].iloc[0]); break
                for col in ['ci_upper', f"{(0.5+level/2)*100:.1f} %", '97.5 %', '97.5%']:
                    if col in ci_df.columns:
                        upper = float(ci_df[col].iloc[0]); break
                if lower is None or upper is None:
                    lower = float(ci_df.iloc[0, 0]); upper = float(ci_df.iloc[0, 1])
                ci = (lower, upper)
        except Exception:
            pass
    if ci is None:
        z = _norm.ppf(0.5 + level/2.0)
        ci = (theta - z*se, theta + z*se)
    return float(theta), float(se), (float(ci[0]), float(ci[1]))


# ---------------- Public API: bias-aware CI and text summaries ----------------

def compute_bias_aware_ci(
    effect_estimation: Dict[str, Any],
    *,
    cf_y: float,
    cf_d: float,
    rho: float = 1.0,
    level: float = 0.95,
    use_signed_rr: bool = False
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - theta, se, level, z
      - sampling_ci
      - theta_bounds_confounding = [theta_lower, theta_upper] = theta ± max_bias
      - bias_aware_ci = [theta - (max_bias + z*se), theta + (max_bias + z*se)]
      - max_bias and components (sigma2, nu2)
    """
    from scipy.stats import norm as _norm
    if not isinstance(effect_estimation, dict) or 'model' not in effect_estimation:
        raise TypeError("Pass the usual result dict with a fitted model under key 'model'.")
    theta, se, sampling_ci = _pull_theta_se_ci(effect_estimation, level)
    z = float(_norm.ppf(0.5 + level/2.0))

    model = effect_estimation['model']
    # Default: no confounding info → bias_aware = sampling CI
    max_bias = 0.0
    sigma2 = np.nan; nu2 = np.nan

    if hasattr(model, "_sensitivity_element_est"):
        elems = model._sensitivity_element_est()
        sigma2 = float(elems["sigma2"])
        psi_sigma2 = np.asarray(elems["psi_sigma2"], float)
        psi_sigma2 = psi_sigma2 - float(np.mean(psi_sigma2))
        m_alpha = np.asarray(elems["m_alpha"], float)
        rr = np.asarray(elems["riesz_rep"], float)
        nu2, psi_nu2 = _combine_nu2_local(m_alpha, rr, cf_y, cf_d, rho, use_signed_rr=use_signed_rr)
        max_bias, _ = _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)

    theta_lower = float(theta) - float(max_bias)
    theta_upper = float(theta) + float(max_bias)
    # Graceful fallback: if se is non-finite, report confounding bounds only
    if not (np.isfinite(se) and se >= 0.0 and np.isfinite(z)):
        bias_aware_ci = (float(theta) - float(max_bias), float(theta) + float(max_bias))
    else:
        bias_aware_ci = (
            float(theta) - (float(max_bias) + z * float(se)),
            float(theta) + (float(max_bias) + z * float(se)),
        )

    return dict(
        theta=float(theta),
        se=float(se),
        level=float(level),
        z=z,
        sampling_ci=tuple(map(float, sampling_ci)),
        theta_bounds_confounding=(float(theta_lower), float(theta_upper)),
        bias_aware_ci=tuple(map(float, bias_aware_ci)),
        max_bias=float(max_bias),
        sigma2=float(sigma2),
        nu2=float(nu2),
        params=dict(cf_y=float(cf_y), cf_d=float(cf_d), rho=float(np.clip(rho, -1.0, 1.0)), use_signed_rr=bool(use_signed_rr)),
    )


def format_bias_aware_summary(res: Dict[str, Any], label: str | None = None) -> str:
    """Pretty, one-row printout (aligned with new summary style)."""
    lbl = (label or 'theta').rjust(6)
    ci_l, ci_u = res['sampling_ci']
    th_l, th_u = res['theta_bounds_confounding']
    bci_l, bci_u = res['bias_aware_ci']
    theta = res['theta']; se = res['se']
    level = res['level']; z = res['z']
    cf = res['params']

    lines = []
    lines.append("================== Bias-aware Interval ==================")
    lines.append("")
    lines.append("------------------ Scenario          ------------------")
    lines.append(f"Significance Level: level={level}")
    lines.append(f"Sensitivity parameters: cf_y={cf['cf_y']}; cf_d={cf['cf_d']}, rho={cf['rho']}, use_signed_rr={cf['use_signed_rr']}")
    lines.append("")
    lines.append("------------------ Components        ------------------")
    lines.append(f"{'':>6} {'theta':>11} {'se':>11} {'z':>8} {'max_bias':>12} {'sigma2':>12} {'nu2':>12}")
    lines.append(f"{lbl} {theta:11.6f} {se:11.6f} {z:8.4f} {res['max_bias']:12.6f} {res['sigma2']:12.6f} {res['nu2']:12.6f}")
    lines.append("")
    lines.append("------------------ Intervals         ------------------")
    lines.append(f"{'':>6} {'Sampling CI lower':>18} {'Conf. θ lower':>16} {'Bias-aware lower':>18} {'Bias-aware upper':>18} {'Conf. θ upper':>16} {'Sampling CI upper':>20}")
    lines.append(f"{lbl} {ci_l:18.6f} {th_l:16.6f} {bci_l:18.6f} {bci_u:18.6f} {th_u:16.6f} {ci_u:20.6f}")
    return "\n".join(lines)


# ---------------- Human-facing wrappers and legacy formatting ----------------

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


def get_sensitivity_summary(
    effect_estimation: Dict[str, Any],
    *,
    label: Optional[str] = None,
) -> Optional[str]:
    """
    Render a single, unified bias-aware summary string.
    If bias-aware components are missing, shows a sampling-only variant with max_bias=0
    and then formats via `format_bias_aware_summary` for consistency.
    """
    if not isinstance(effect_estimation, dict) or 'model' not in effect_estimation:
        return None

    model = effect_estimation['model']
    if label is None:
        t = getattr(getattr(model, 'data', None), 'treatment', None)
        label = getattr(t, 'name', None) or 'theta'

    res = effect_estimation.get('bias_aware')

    # Build a sampling-only placeholder if needed (level fixed at 0.95 here)
    if not isinstance(res, dict):
        theta, se, ci = _pull_theta_se_ci(effect_estimation, level=0.95)
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + 0.95 / 2.0))
        res = dict(
            theta=float(theta),
            se=float(se),
            level=0.95,
            z=z,
            sampling_ci=(float(ci[0]), float(ci[1])),
            theta_bounds_confounding=(float(theta), float(theta)),  # max_bias = 0
            bias_aware_ci=(float(theta - z * se), float(theta + z * se)),
            max_bias=0.0,
            sigma2=np.nan,
            nu2=np.nan,
            params=dict(cf_y=0.0, cf_d=0.0, rho=0.0, use_signed_rr=False),
        )

    # Single clean summary (reuse the one definitive formatter)
    return format_bias_aware_summary(res, label=label)


# ---------------- Benchmarking sensitivity (short vs long model) ----------------

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
        from causalis.data.causaldata import CausalData
        from causalis.inference.estimators.irm import IRM

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

    def _center_w(a: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, float)
        a = np.asarray(a, float)
        sw = float(np.sum(w))
        mu = float(np.sum(w * a)) / (sw if sw > 1e-12 else 1.0)
        return a - mu

    def _ols_r2_and_fit(yv: np.ndarray, Z: np.ndarray, w: Optional[np.ndarray] = None) -> tuple[float, np.ndarray]:
        """Stable (weighted) OLS on centered & standardized vars for R^2 and fitted component."""
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if w is None:
            # Unweighted
            yv_c = _center(yv)
            Zc = Z - np.nanmean(Z, axis=0, keepdims=True)
            col_std = np.nanstd(Zc, axis=0, ddof=0)
            valid = np.isfinite(col_std) & (col_std > 1e-12)
            if not np.any(valid):
                return 0.0, np.zeros_like(yv_c)
            Zcs = Zc[:, valid] / col_std[valid]
            Zcs = np.nan_to_num(Zcs, nan=0.0, posinf=0.0, neginf=0.0)
            yv_c = np.nan_to_num(np.asarray(yv_c, float), nan=0.0, posinf=0.0, neginf=0.0)
            from numpy.linalg import lstsq
            beta, *_ = lstsq(Zcs, yv_c, rcond=1e-12)
            yhat = Zcs @ beta
            denom = float(np.dot(yv_c, yv_c))
            if not np.isfinite(denom) or denom <= 1e-12:
                return 0.0, np.zeros_like(yv_c)
            num = float(np.dot(yhat, yhat))
            if not np.isfinite(num) or num < 0.0:
                return 0.0, np.zeros_like(yv_c)
            r2 = float(np.clip(num / denom, 0.0, 1.0))
            return r2, yhat
        else:
            # Weighted
            w = np.asarray(w, float)
            # sanitize weights and features to avoid NaN/inf propagation
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            Z = np.asarray(Z, float)
            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            sw = float(np.sum(w))
            if not np.isfinite(sw) or sw <= 1e-12:
                return 0.0, np.zeros_like(yv, dtype=float)
            yv_c = _center_w(yv, w)
            # Weighted column means
            muZ = (w[:, None] * Z).sum(axis=0) / sw
            Zc = Z - muZ
            # Weighted std per column
            var = (w[:, None] * (Zc * Zc)).sum(axis=0) / sw
            std = np.sqrt(np.maximum(var, 0.0))
            valid = np.isfinite(std) & (std > 1e-12)
            if not np.any(valid):
                return 0.0, np.zeros_like(yv_c)
            Zcs = Zc[:, valid] / std[valid]
            Zcs = np.nan_to_num(Zcs, nan=0.0, posinf=0.0, neginf=0.0)
            yv_c = np.nan_to_num(np.asarray(yv_c, float), nan=0.0, posinf=0.0, neginf=0.0)
            # Weighted least squares via sqrt(w)
            swr = np.sqrt(np.clip(w, 0.0, np.inf))
            Zsw = Zcs * swr[:, None]
            ysw = yv_c * swr
            from numpy.linalg import lstsq
            beta, *_ = lstsq(Zsw, ysw, rcond=1e-12)
            yhat = Zcs @ beta
            denom = float(np.dot(ysw, ysw))
            if not np.isfinite(denom) or denom <= 1e-12:
                return 0.0, np.zeros_like(yv_c)
            num = float(np.dot(swr * yhat, swr * yhat))
            if not np.isfinite(num) or num < 0.0:
                return 0.0, np.zeros_like(yv_c)
            r2 = float(np.clip(num / denom, 0.0, 1.0))
            return r2, yhat

    Z = df[benchmarking_set].to_numpy(dtype=float)
    # ATT weighting if applicable
    p = float(np.mean(d)) if (np.isfinite(np.mean(d)) and np.mean(d) > 0.0) else 1.0
    w_att = np.where(d > 0.5, 1.0 / max(p, 1e-12), 0.0)
    is_att = str(getattr(model, 'score', '')).upper().startswith('ATT')
    weights = w_att if is_att else None

    R2y, yhat_u = _ols_r2_and_fit(r_y, Z, w=weights)
    R2d, dhat_u = _ols_r2_and_fit(r_d, Z, w=weights)
    cf_y = float(R2y / max(1e-12, 1.0 - R2y))
    cf_d = float(R2d / max(1e-12, 1.0 - R2d))

    def _safe_corr(u: np.ndarray, v: np.ndarray, w: Optional[np.ndarray] = None) -> float:
        if w is None:
            u = _center(u); v = _center(v)
            su, sv = np.std(u), np.std(v)
            if not (np.isfinite(su) and np.isfinite(sv)) or su <= 0 or sv <= 0:
                return 0.0
            val = float(np.corrcoef(u, v)[0, 1])
            return float(np.clip(val, -1.0, 1.0))
        u = _center_w(u, w); v = _center_w(v, w)
        sw = float(np.sum(w))
        su = np.sqrt(max(0.0, float(np.sum(w * u * u)) / (sw if sw > 1e-12 else 1.0)))
        sv = np.sqrt(max(0.0, float(np.sum(w * v * v)) / (sw if sw > 1e-12 else 1.0)))
        if su <= 0 or sv <= 0:
            return 0.0
        cov = float(np.sum(w * u * v)) / (sw if sw > 1e-12 else 1.0)
        val = cov / (su * sv)
        return float(np.clip(val, -1.0, 1.0))

    rho = _safe_corr(yhat_u, dhat_u, w=weights)

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


# ---------------- Main entry for producing textual sensitivity summary ----------------

def sensitivity_analysis(
    effect_estimation: Dict[str, Any],
    *,
    cf_y: float,
    cf_d: float,
    rho: float = 1.0,
    level: float = 0.95,
    use_signed_rr: bool = False,
) -> Dict[str, Any]:
    """
    Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.

    Returns a dict with:
      - theta, se, level, z
      - sampling_ci
      - theta_bounds_confounding = (theta - max_bias, theta + max_bias)
      - bias_aware_ci = [theta - (max_bias + z*se), theta + (max_bias + z*se)]
      - max_bias and components (sigma2, nu2)
      - params (cf_y, cf_d, rho, use_signed_rr)
    """
    if not isinstance(effect_estimation, dict) or 'model' not in effect_estimation:
        raise TypeError("Pass a result dict with a fitted model under key 'model'.")
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be in (0,1).")
    if cf_y < 0 or cf_d < 0:
        raise ValueError("cf_y and cf_d must be >= 0.")

    from scipy.stats import norm as _norm

    theta, se, sampling_ci = _pull_theta_se_ci(effect_estimation, level)
    z = float(_norm.ppf(0.5 + level / 2.0))

    model = effect_estimation['model']
    max_bias = 0.0
    sigma2 = np.nan
    nu2 = np.nan

    if hasattr(model, "_sensitivity_element_est"):
        elems = model._sensitivity_element_est()
        sigma2 = float(elems["sigma2"])
        psi_sigma2 = np.asarray(elems["psi_sigma2"], float)
        psi_sigma2 = psi_sigma2 - float(np.mean(psi_sigma2))
        m_alpha = np.asarray(elems["m_alpha"], float)
        rr = np.asarray(elems["riesz_rep"], float)
        nu2, psi_nu2 = _combine_nu2_local(
            m_alpha, rr, cf_y=cf_y, cf_d=cf_d, rho=rho, use_signed_rr=use_signed_rr
        )
        max_bias, _ = _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)

    theta_lower = float(theta) - float(max_bias)
    theta_upper = float(theta) + float(max_bias)
    # Graceful fallback: if se is non-finite, report confounding bounds only
    if not (np.isfinite(se) and se >= 0.0 and np.isfinite(z)):
        bias_aware_ci = (float(theta) - float(max_bias), float(theta) + float(max_bias))
    else:
        bias_aware_ci = (
            float(theta) - (float(max_bias) + z * float(se)),
            float(theta) + (float(max_bias) + z * float(se)),
        )

    res = dict(
        theta=float(theta),
        se=float(se),
        level=float(level),
        z=z,
        sampling_ci=tuple(map(float, sampling_ci)),
        theta_bounds_confounding=(float(theta_lower), float(theta_upper)),
        bias_aware_ci=tuple(map(float, bias_aware_ci)),
        max_bias=float(max_bias),
        sigma2=float(sigma2),
        nu2=float(nu2),
        params=dict(
            cf_y=float(cf_y),
            cf_d=float(cf_d),
            rho=float(np.clip(rho, -1.0, 1.0)),
            use_signed_rr=bool(use_signed_rr),
        ),
    )

    effect_estimation["bias_aware"] = res
    return res
