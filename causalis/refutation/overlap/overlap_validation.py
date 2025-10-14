from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union

# ---------- small utils ----------


def _mask_finite_pairs(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pairs (p,y) keeping only entries where both are finite."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if p.size == 0 or y.size == 0 or p.size != y.size:
        return p, y
    mask = np.isfinite(p) & np.isfinite(y)
    return p[mask], y[mask]

def _auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Dependency-free AUC via Mann–Whitney U."""
    y = labels.astype(bool)
    pos = scores[y]
    neg = scores[~y]
    n1, n0 = pos.size, neg.size
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")  # stable
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            # average of ranks in this tied block (1-based)
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    R1 = ranks[y].sum()
    auc = (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic (D) that handles ties correctly.

    Computes the sup norm of the difference between empirical CDFs,
    evaluating at the union of unique values with right-continuous CDFs.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return float("nan")
    vals = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, vals, side="right") / na
    cdf_b = np.searchsorted(b, vals, side="right") / nb
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ess(w: np.ndarray) -> float:
    s = w.sum()
    q = (w ** 2).sum()
    return float((s * s) / q) if q > 0 else float("nan")


def _q(w: np.ndarray, qs=(0.5, 0.95, 0.99, 0.999)) -> Dict[str, float]:
    if w.size == 0:
        return {f"q{int(100*q)}": float("nan") for q in qs} | {"max": float("nan")}
    # Prefer smoother linear interpolation for quantiles; fallback for older NumPy
    try:
        quant = np.quantile(w, qs, method="linear")
    except TypeError:
        quant = np.quantile(w, qs)
    return {f"q{int(100*q)}": float(v) for q, v in zip(qs, quant)} | {"max": float(np.max(w))}


# ---------- thresholds you can tweak ----------

DEFAULT_THRESHOLDS = dict(
    edge_mass_warn_001=0.02, edge_mass_strong_001=0.05,
    edge_mass_warn_002=0.05, edge_mass_strong_002=0.10,
    ks_warn=0.25, ks_strong=0.35,
    auc_warn=0.90, auc_strong=0.95,
    ipw_relerr_warn=0.05, ipw_relerr_strong=0.10,
    ess_ratio_warn=0.30, ess_ratio_strong=0.15,
    clip_share_warn=0.02, clip_share_strong=0.05,
    tail_vs_med_warn=10.0
)




# ---------- split diagnostics API ----------

def edge_mass(m_hat: np.ndarray, eps: Union[float, Tuple[float, ...], list, np.ndarray] = 0.01) -> Dict[Any, Any]:
    """
    Edge mass diagnostics.

    Math:
      share_below = (1/n) * sum_i 1{ m_hat_i < ε }
      share_above = (1/n) * sum_i 1{ m_hat_i > 1 - ε }

    Parameters:
      - m_hat: array of propensities m_hat in [0,1]
      - eps: a single ε or a sequence of ε values.

    Returns:
      - If eps is a scalar: {'eps': ε, 'share_below': float, 'share_above': float}
      - If eps is a sequence: {ε: {'share_below': float, 'share_above': float}, ...}
    """
    m = np.asarray(m_hat, dtype=float)
    if np.isscalar(eps):
        e = float(eps)
        return {
            'eps': e,
            'share_below': float(np.mean(m < e)) if m.size else float('nan'),
            'share_above': float(np.mean(m > 1.0 - e)) if m.size else float('nan'),
        }
    out: Dict[Any, Any] = {}
    for e in list(eps):
        e = float(e)
        out[e] = {
            'share_below': float(np.mean(m < e)) if m.size else float('nan'),
            'share_above': float(np.mean(m > 1.0 - e)) if m.size else float('nan'),
        }
    return out


def ks_distance(m_hat: np.ndarray, D: np.ndarray) -> float:
    """
    Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.

    Math:
      KS = sup_t | F_{m|D=1}(t) - F_{m|D=0}(t) |
    """
    m = np.asarray(m_hat, dtype=float).ravel()
    d = np.asarray(D, dtype=int).ravel().astype(bool)
    # scrub non-finite propensities
    mask = np.isfinite(m)
    if mask.size != m.size:
        # defensive, but proceed
        pass
    m = m[mask]
    d = d[mask]
    if m.size == 0 or d.sum() == 0 or d.sum() == d.size:
        return float('nan')
    return _ks_statistic(m[d], m[~d])


def auc_for_m(m_hat: np.ndarray, D: np.ndarray) -> float:
    """
    ROC AUC using scores m_hat vs labels D.

    Math (Mann–Whitney relation):
      AUC = P(m_i^+ > m_j^-) + 0.5 P(m_i^+ = m_j^-)
    """
    m = np.asarray(m_hat, dtype=float).ravel()
    d = np.asarray(D, dtype=int).ravel()
    if m.size == 0 or d.size == 0 or m.size != d.size:
        return float('nan')
    # scrub non-finite pairs
    m, d_f = _mask_finite_pairs(m, d)
    d = d_f.astype(int)
    if m.size == 0 or np.unique(d).size < 2:
        return float('nan')
    return _auc_mann_whitney(m, d)


def ess_per_group(m_hat: np.ndarray, D: np.ndarray) -> Dict[str, float]:
    """
    Effective sample size (ESS) for ATE-style inverse-probability weights per arm.

    Weights:
      w1_i = D_i / m_hat_i,
      w0_i = (1 - D_i) / (1 - m_hat_i).

    ESS:
      ESS(w_g) = (sum_i w_{gi})^2 / sum_i w_{gi}^2.

    Returns dict with ess and ratios (ESS / group size).
    """
    m = np.asarray(m_hat, dtype=float)
    d = np.asarray(D, dtype=int).astype(bool)
    if m.size == 0:
        return {
            'ess_w1': float('nan'), 'ess_w0': float('nan'),
            'ess_ratio_w1': float('nan'), 'ess_ratio_w0': float('nan')
        }
    m_safe = np.clip(m, 1e-12, 1 - 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        w1 = np.where(d, 1.0 / m_safe, 0.0)
        w0 = np.where(~d, 1.0 / (1.0 - m_safe), 0.0)
    n1 = int(d.sum())
    n0 = int((~d).sum())
    e1 = _ess(w1[d]) if n1 else float('nan')
    e0 = _ess(w0[~d]) if n0 else float('nan')
    return {
        'ess_w1': float(e1),
        'ess_w0': float(e0),
        'ess_ratio_w1': float(e1 / n1) if n1 else float('nan'),
        'ess_ratio_w0': float(e0 / n0) if n0 else float('nan'),
    }


def att_weight_sum_identity(m_hat: np.ndarray, D: np.ndarray) -> Dict[str, float]:
    """
    ATT weight-sum identity check (un-normalized IPW form).

    Math:
      w1_i = D_i / p1,   w0_i = (1 - D_i) * m_hat_i / ((1 - m_hat_i) * p1),  where p1 = (1/n) sum_i D_i.
      Sum check:  sum_i (1 - D_i) * m_hat_i / (1 - m_hat_i)  ?≈  sum_i D_i.

    Returns: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float}
    """
    m = np.asarray(m_hat, dtype=float)
    d = np.asarray(D, dtype=int)
    if m.size == 0:
        return {'lhs_sum': float('nan'), 'rhs_sum': float('nan'), 'rel_err': float('nan')}
    m_safe = np.clip(m, 1e-12, 1 - 1e-12)
    lhs = float(np.sum((1 - d.astype(float)) * (m_safe / (1.0 - m_safe))))
    rhs = float(np.sum(d))
    rel_err = abs(lhs - rhs) / rhs if rhs > 0 else float('inf')
    return {'lhs_sum': lhs, 'rhs_sum': rhs, 'rel_err': float(rel_err)}


# ---------- helpers for composed checks ----------

def _edge_mass_by_arm(m_hat: np.ndarray, D: np.ndarray, eps: Tuple[float, float] = (0.01, 0.02)) -> Dict[str, float]:
    m = np.asarray(m_hat, dtype=float)
    d = np.asarray(D, dtype=int).astype(bool)
    n = m.size
    n1 = int(d.sum())
    e1, e2 = float(eps[0]), float(eps[1])

    def _mean_mask(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if mask.size else float("nan")

    return {
        "share_below_001_D1": _mean_mask(m[d] < e1) if n1 else float("nan"),
        "share_above_001_D0": _mean_mask(m[~d] > 1 - e1) if n1 < n else float("nan"),
        "share_below_002_D1": _mean_mask(m[d] < e2) if n1 else float("nan"),
        "share_above_002_D0": _mean_mask(m[~d] > 1 - e2) if n1 < n else float("nan"),
    }


def _ate_weights(m_hat: np.ndarray, D: np.ndarray, use_hajek: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    m = np.clip(np.asarray(m_hat, dtype=float), 1e-12, 1.0 - 1e-12)
    d = np.asarray(D, dtype=int).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.where(d, 1.0 / m, 0.0)
        w0 = np.where(~d, 1.0 / (1.0 - m), 0.0)
    if use_hajek:
        n1 = int(d.sum()); n0 = int((~d).sum())
        if n1:
            w1 = w1 / np.mean(w1[d])
        if n0:
            w0 = w0 / np.mean(w0[~d])
    return w1, w0


def _ate_ipw_stats_from_weights(w1: np.ndarray, w0: np.ndarray, D: np.ndarray, use_hajek: bool = False) -> Dict[str, float]:
    d = np.asarray(D, dtype=int).astype(bool)
    n = d.size
    n1 = int(d.sum())
    return dict(
        sum_w1=float(np.sum(w1)),
        sum_w0=float(np.sum(w0)),
        mean_w1=float(np.mean(w1[d])) if n1 else float("nan"),
        mean_w0=float(np.mean(w0[~d])) if n1 < n else float("nan"),
        target_sum_w1=float(n1),
        target_sum_w0=float(n - n1),
        target_mean_w=1.0 if use_hajek else float("nan"),
    )


def _ate_ess_from_weights(w1: np.ndarray, w0: np.ndarray, D: np.ndarray) -> Dict[str, float]:
    d = np.asarray(D, dtype=int).astype(bool)
    n = d.size
    n1 = int(d.sum())
    return dict(
        ess_w1=float(_ess(w1[d])) if n1 else float("nan"),
        ess_w0=float(_ess(w0[~d])) if n1 < n else float("nan"),
        ess_ratio_w1=float(_ess(w1[d]) / n1) if n1 else float("nan"),
        ess_ratio_w0=float(_ess(w0[~d]) / (n - n1)) if n1 < n else float("nan"),
    )


def _ate_tails_from_weights(w1: np.ndarray, w0: np.ndarray, D: np.ndarray) -> Dict[str, Dict[str, float]]:
    d = np.asarray(D, dtype=int).astype(bool)
    n = d.size
    n1 = int(d.sum())
    med_w1 = float(np.median(w1[d])) if n1 else float("nan")
    med_w0 = float(np.median(w0[~d])) if n1 < n else float("nan")
    return dict(
        w1=_q(w1[d]) | {"median": med_w1},
        w0=_q(w0[~d]) | {"median": med_w0},
    )


def _clipping_audit(m_hat: np.ndarray, m_clipped_from: Optional[Tuple[float, float]], g_clipped_share: Optional[float]) -> Dict[str, float]:
    if m_clipped_from is None:
        clip_lower = float("nan")
        clip_upper = float("nan")
    else:
        lo, hi = m_clipped_from
        m = np.asarray(m_hat, dtype=float)
        clip_lower = float(np.mean(m <= lo))
        clip_upper = float(np.mean(m >= hi))
    return dict(
        m_clip_lower=clip_lower,
        m_clip_upper=clip_upper,
        g_clip_share=float(g_clipped_share) if g_clipped_share is not None else float("nan"),
    )


# ---------- main entry ----------

# ---------- Propensity calibration (merged from propensity_calibration.py) ----------
CAL_THRESHOLDS = dict(
    ece_warn=0.10,
    ece_strong=0.20,
    slope_warn_lo=0.8,
    slope_warn_hi=1.2,
    slope_strong_lo=0.6,
    slope_strong_hi=1.4,
    intercept_warn=0.2,
    intercept_strong=0.4,
)


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].

    Parameters
    ----------
    p : np.ndarray
        Predicted probabilities in [0,1]. Will be clipped to [0,1].
    y : np.ndarray
        Binary labels {0,1}.
    n_bins : int, default 10
        Number of bins.

    Returns
    -------
    float
        ECE value in [0,1].
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    n_bins = int(n_bins)
    if p.size == 0 or y.size == 0 or p.size != y.size:
        return float("nan")

    # Bin indices consistent with equal-width bins on [0,1]
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    sums = np.bincount(b, weights=p, minlength=n_bins)
    hits = np.bincount(b, weights=y, minlength=n_bins)
    cnts = np.bincount(b, minlength=n_bins).astype(float)
    mask = cnts > 0
    if not np.any(mask):
        return float("nan")
    return float(
        np.average(
            np.abs(hits[mask] / cnts[mask] - sums[mask] / cnts[mask]),
            weights=cnts[mask] / cnts[mask].sum(),
        )
    )


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    return np.log(x / (1.0 - x))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    # For positive z: exp(-z) is safe
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    # For negative z: use exp(z) to avoid overflow
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


def _logistic_recalibration(p: np.ndarray, y: np.ndarray, *, max_iter: int = 100, tol: float = 1e-8, ridge: float = 1e-8) -> tuple[float, float]:
    """
    Fit logistic recalibration model: Pr(D=1|p) = sigmoid(alpha + beta * logit(p)).
    Uses IRLS/Newton steps with numerical guards to avoid over/underflow and singularities.

    Returns (alpha, beta).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Predictor on log-odds scale
    z = _logit(p)

    # Handle near-constant predictor: no meaningful slope can be estimated
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))
    pi = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    base_intercept = float(np.log(pi / (1.0 - pi)))
    if not np.isfinite(z_std) or z_std < 1e-12:
        # Intercept-only calibration
        return base_intercept, 0.0

    # Standardize to improve conditioning, then back-transform at the end
    z_stdized = (z - z_mean) / z_std
    Xs = np.column_stack([np.ones_like(z_stdized), z_stdized])  # [1, z_std]

    theta = np.array([base_intercept, 1.0], dtype=float)

    for _ in range(max_iter):
        # Linear predictor with clipping to avoid extreme magnitudes
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            eta = Xs @ theta
        # Bound eta before sigmoid to avoid exp overflow inside sigmoid
        eta = np.clip(eta, -40.0, 40.0)
        mu = _sigmoid(eta)
        # Guard against degenerate probabilities
        mu = np.clip(mu, 1e-12, 1.0 - 1e-12)

        # Gradient and Hessian
        r = y - mu  # residuals
        W = mu * (1.0 - mu)
        # Build X^T W X and X^T r
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            XT_W = Xs.T * W  # broadcasting
            H = XT_W @ Xs
            g = Xs.T @ r
        # Ridge for stability
        H[0, 0] += ridge
        H[1, 1] += ridge
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            step = np.linalg.pinv(H) @ g

        # Dampen excessively large steps to prevent parameter blow-up
        max_step = np.max(np.abs(step))
        if not np.isfinite(max_step):
            break
        if max_step > 5.0:
            step = step * (5.0 / max_step)

        theta_new = theta + step
        if np.max(np.abs(step)) < tol:
            theta = theta_new
            break
        # Bound parameters to a reasonable range to keep X@theta finite
        theta = np.clip(theta_new, -50.0, 50.0)

    # Back-transform to original scale: z = z_mean + z_std * z_stdized
    beta = float(theta[1] / z_std)
    alpha = float(theta[0] - theta[1] * (z_mean / z_std))

    # Final sanity: ensure finite outputs
    if not np.isfinite(alpha):
        alpha = base_intercept
    if not np.isfinite(beta):
        beta = 0.0
    return alpha, beta


def calibration_report_m(
    m_hat: np.ndarray,
    D: np.ndarray,
    n_bins: int = 10,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Propensity calibration report for cross-fitted propensities m_hat against treatment D.

    Returns a dictionary with:
      - auc: ROC AUC of m_hat vs D (Mann–Whitney)
      - brier: Brier score (mean squared error)
      - ece: Expected Calibration Error (equal-width bins)
      - reliability_table: pd.DataFrame with per-bin stats
      - recalibration: {'intercept': alpha, 'slope': beta} from logistic recalibration
      - flags: {'ece': ..., 'slope': ..., 'intercept': ...} using GREEN/YELLOW/RED
    """
    p = np.asarray(m_hat, dtype=float).ravel()
    y = np.asarray(D, dtype=int).ravel()
    if p.size == 0 or y.size == 0 or p.size != y.size:
        raise ValueError("m_hat and D must be non-empty arrays of the same length")

    # Scrub non-finite pairs before clipping
    p, y_f = _mask_finite_pairs(p, y)
    y = y_f.astype(int)
    if p.size == 0 or y.size == 0 or p.size != y.size:
        raise ValueError("m_hat and D must be non-empty finite arrays of the same length")

    # Clip probabilities to avoid infinities and invalid ops
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    # Metrics
    auc = float(_auc_mann_whitney(p, y)) if np.unique(y).size == 2 else float("nan")
    brier = float(np.mean((p - y) ** 2))
    ece = float(ece_binary(p, y, n_bins=n_bins))

    # Reliability table using same binning as ECE
    n_bins = int(n_bins)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    cnts = np.bincount(b, minlength=n_bins).astype(int)
    sum_p = np.bincount(b, weights=p, minlength=n_bins)
    sum_y = np.bincount(b, weights=y, minlength=n_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_p = np.where(cnts > 0, sum_p / cnts, np.nan)
        frac_pos = np.where(cnts > 0, sum_y / cnts, np.nan)
        abs_err = np.abs(frac_pos - mean_p)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rel_df = pd.DataFrame({
        "bin": np.arange(n_bins),
        "lower": edges[:-1],
        "upper": edges[1:],
        "count": cnts,
        "mean_p": mean_p,
        "frac_pos": frac_pos,
        "abs_error": abs_err,
    })

    # Logistic recalibration
    alpha, beta = _logistic_recalibration(p, y)

    thr = CAL_THRESHOLDS.copy()
    if isinstance(thresholds, dict):
        thr.update({k: float(v) for k, v in thresholds.items()})

    # Flags
    def _flag_ece(val: float) -> str:
        if np.isnan(val):
            return "NA"
        if val > thr["ece_strong"]:
            return "RED"
        if val > thr["ece_warn"]:
            return "YELLOW"
        return "GREEN"

    def _flag_slope(b: float) -> str:
        if np.isnan(b):
            return "NA"
        if b < thr["slope_strong_lo"] or b > thr["slope_strong_hi"]:
            return "RED"
        if b < thr["slope_warn_lo"] or b > thr["slope_warn_hi"]:
            return "YELLOW"
        return "GREEN"

    def _flag_intercept(a: float) -> str:
        if np.isnan(a):
            return "NA"
        if abs(a) > thr["intercept_strong"]:
            return "RED"
        if abs(a) > thr["intercept_warn"]:
            return "YELLOW"
        return "GREEN"

    flags = {
        "ece": _flag_ece(ece),
        "slope": _flag_slope(beta),
        "intercept": _flag_intercept(alpha),
    }

    return {
        "n": int(p.size),
        "n_bins": int(n_bins),
        "auc": auc,
        "brier": brier,
        "ece": ece,
        "reliability_table": rel_df,
        "recalibration": {"intercept": float(alpha), "slope": float(beta)},
        "flags": flags,
        "thresholds": thr,
    }

def positivity_overlap_checks(
    m_hat: np.ndarray,
    D: np.ndarray,
    *,
    m_clipped_from: Optional[Tuple[float, float]] = None,
    g_clipped_share: Optional[float] = None,
    use_hajek: bool = False,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    n_bins: int = 10,
    cal_thresholds: Optional[Dict[str, float]] = None,
    auc_flip_margin: float = 0.05,
) -> Dict[str, Any]:
    """
    Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
    Inputs are cross-fitted m̂ and treatment D (0/1). Returns a structured report with GREEN/YELLOW/RED flags.
    """
    m_hat = np.asarray(m_hat, dtype=float)
    D = np.asarray(D, dtype=int).astype(bool)
    n = m_hat.size
    n1 = int(D.sum())
    p1 = n1 / n if n > 0 else float("nan")

    # Edge mass, including by arm
    eps1, eps2 = 0.01, 0.02
    em_map = edge_mass(m_hat, eps=(eps1, eps2))
    edge_mass_stats = {
        "share_below_001": float(em_map[eps1]["share_below"]) if n else float("nan"),
        "share_above_001": float(em_map[eps1]["share_above"]) if n else float("nan"),
        "share_below_002": float(em_map[eps2]["share_below"]) if n else float("nan"),
        "share_above_002": float(em_map[eps2]["share_above"]) if n else float("nan"),
        "min_m": float(np.min(m_hat)) if n else float("nan"),
        "max_m": float(np.max(m_hat)) if n else float("nan"),
    }

    edge_mass_by_arm = _edge_mass_by_arm(m_hat, D, (eps1, eps2))

    # KS & AUC
    ks = ks_distance(m_hat, D) if (n1 and n1 < n) else float("nan")
    auc = auc_for_m(m_hat, D.astype(int)) if (n1 and n1 < n) else float("nan")

    # ATE weights-based checks
    w1, w0 = _ate_weights(m_hat, D, use_hajek=use_hajek)
    ate_ipw = _ate_ipw_stats_from_weights(w1, w0, D, use_hajek=use_hajek)
    ate_ess = _ate_ess_from_weights(w1, w0, D)
    ate_tails = _ate_tails_from_weights(w1, w0, D)

    # ATT weights and identity (unified via helper)
    if p1 in (0.0, 1.0) or np.isnan(p1):
        att_weights = dict(lhs_sum=float("nan"), rhs_sum=float("nan"), rel_err=float("nan"))
        att_ess = dict(ess_w1=float("nan"), ess_w0=float("nan"), ess_ratio_w1=float("nan"), ess_ratio_w0=float("nan"))
    else:
        att_id = att_weight_sum_identity(m_hat, D.astype(int))
        att_weights = dict(lhs_sum=att_id["lhs_sum"], rhs_sum=att_id["rhs_sum"], rel_err=att_id["rel_err"])
        m_safe = np.clip(m_hat.astype(float), 1e-12, 1.0 - 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            w1_att = D.astype(float) / p1
            w0_att = ((~D).astype(float) * (m_safe / (1.0 - m_safe))) / p1
        n0 = int((~D).sum())
        att_ess = dict(
            ess_w1=float(_ess(w1_att[D])) if n1 else float("nan"),
            ess_w0=float(_ess(w0_att[~D])) if n0 else float("nan"),
            ess_ratio_w1=float(_ess(w1_att[D]) / n1) if n1 else float("nan"),
            ess_ratio_w0=float(_ess(w0_att[~D]) / n0) if n0 else float("nan"),
        )

    # clipping audit
    clipping = _clipping_audit(m_hat, m_clipped_from, g_clipped_share)

    # Flags
    flags: Dict[str, str] = {}

    # edge mass flags
    em = edge_mass_stats
    if em["share_below_001"] > thresholds["edge_mass_strong_001"] or em["share_above_001"] > thresholds["edge_mass_strong_001"]:
        flags["edge_mass_001"] = "RED"
    elif em["share_below_001"] > thresholds["edge_mass_warn_001"] or em["share_above_001"] > thresholds["edge_mass_warn_001"]:
        flags["edge_mass_001"] = "YELLOW"
    else:
        flags["edge_mass_001"] = "GREEN"

    if em["share_below_002"] > thresholds["edge_mass_strong_002"] or em["share_above_002"] > thresholds["edge_mass_strong_002"]:
        flags["edge_mass_002"] = "RED"
    elif em["share_below_002"] > thresholds["edge_mass_warn_002"] or em["share_above_002"] > thresholds["edge_mass_warn_002"]:
        flags["edge_mass_002"] = "YELLOW"
    else:
        flags["edge_mass_002"] = "GREEN"

    # KS & AUC flags
    if not np.isnan(ks):
        flags["ks"] = "RED" if ks > thresholds["ks_strong"] else ("YELLOW" if ks > thresholds["ks_warn"] else "GREEN")
    else:
        flags["ks"] = "NA"
    if not np.isnan(auc):
        sep = max(auc, 1.0 - auc)
        flags["auc"] = "RED" if sep > thresholds["auc_strong"] else ("YELLOW" if sep > thresholds["auc_warn"] else "GREEN")
        # flip hint if AUC noticeably below 0.5
        try:
            margin = float(auc_flip_margin)
        except Exception:
            margin = 0.05
        if auc < (0.5 - margin):
            flags["auc_flip_suspected"] = "YELLOW"
    else:
        flags["auc"] = "NA"

    # IPW sum checks (ATE): only meaningful under Hájek normalization
    def _relerr(a: float, b: float) -> float:
        if np.isnan(a) or np.isnan(b) or b == 0:
            return float("nan")
        return abs(a - b) / abs(b)

    if use_hajek:
        relerr_w1 = _relerr(ate_ipw["sum_w1"], ate_ipw["target_sum_w1"])
        relerr_w0 = _relerr(ate_ipw["sum_w0"], ate_ipw["target_sum_w0"])
        for name, re in (("ipw_sum_w1", relerr_w1), ("ipw_sum_w0", relerr_w0)):
            if np.isnan(re):
                flags[name] = "NA"
            elif re > thresholds["ipw_relerr_strong"]:
                flags[name] = "RED"
            elif re > thresholds["ipw_relerr_warn"]:
                flags[name] = "YELLOW"
            else:
                flags[name] = "GREEN"
    else:
        # Under Horvitz–Thompson (non-Hájek) normalization, these sums don't have a fixed target per arm.
        flags["ipw_sum_w1"] = "NA"
        flags["ipw_sum_w0"] = "NA"

    # ESS flags (ATE)
    for side in ("w1", "w0"):
        ratio = ate_ess.get(f"ess_ratio_{side}", float("nan"))
        if np.isnan(ratio):
            flags[f"ess_{side}"] = "NA"
        elif ratio < thresholds["ess_ratio_strong"]:
            flags[f"ess_{side}"] = "RED"
        elif ratio < thresholds["ess_ratio_warn"]:
            flags[f"ess_{side}"] = "YELLOW"
        else:
            flags[f"ess_{side}"] = "GREEN"

    # Tail vs median flags
    for side in ("w1", "w0"):
        tails = ate_tails[side]
        med = tails.get("median", float("nan"))
        if np.isnan(med) or med == 0:
            flags[f"tails_{side}"] = "NA"
        else:
            blowups = []
            for k in ("q95", "q99", "q999", "max"):
                v = tails.get(k, float("nan"))
                if not np.isnan(v):
                    blowups.append(v / med)
            big_yellow = any(b > thresholds["tail_vs_med_warn"] for b in blowups)
            big_red = any(b > 100 for b in blowups)
            flags[f"tails_{side}"] = "RED" if big_red else ("YELLOW" if big_yellow else "GREEN")

    # ATT identity
    rhs_sum = att_weights.get("rhs_sum", float("nan")) if isinstance(att_weights, dict) else float("nan")
    rel_err_val = att_weights.get("rel_err", float("nan")) if isinstance(att_weights, dict) else float("nan")
    if np.isfinite(rhs_sum) and rhs_sum > 0 and np.isfinite(rel_err_val):
        re = float(rel_err_val)
        flags["att_identity"] = (
            "RED" if re > thresholds["ipw_relerr_strong"] else ("YELLOW" if re > thresholds["ipw_relerr_warn"] else "GREEN")
        )
    else:
        flags["att_identity"] = "NA"

    # Clipping flags
    clip_lower = clipping.get("m_clip_lower", float("nan"))
    clip_upper = clipping.get("m_clip_upper", float("nan"))
    if np.isnan(clip_lower) or np.isnan(clip_upper):
        flags["clip_m"] = "NA"
    else:
        clip_total = clip_lower + clip_upper
        if clip_total > thresholds["clip_share_strong"]:
            flags["clip_m"] = "RED"
        elif clip_total > thresholds["clip_share_warn"]:
            flags["clip_m"] = "YELLOW"
        else:
            flags["clip_m"] = "GREEN"

    # --- Propensity calibration (merged) ---
    try:
        cal_report = calibration_report_m(m_hat, D.astype(int), n_bins=int(n_bins), thresholds=cal_thresholds)
        # Merge calibration flags with clear prefixes to avoid collisions
        cal_flags = cal_report.get("flags", {})
        flags["calibration_ece"] = cal_flags.get("ece", "NA")
        flags["calibration_slope"] = cal_flags.get("slope", "NA")
        flags["calibration_intercept"] = cal_flags.get("intercept", "NA")
    except Exception:
        cal_report = None

    return dict(
        n=int(n),
        n_treated=int(n1),
        p1=float(p1),
        eps=(float(eps1), float(eps2)),
        edge_mass=edge_mass_stats,
        edge_mass_by_arm=edge_mass_by_arm,
        ks=float(ks),
        auc=float(auc),
        ate_ipw=ate_ipw,
        ate_ess=ate_ess,
        ate_tails=ate_tails,
        att_weights=att_weights,
        att_ess=att_ess,
        clipping=clipping,
        calibration=cal_report,
        flags=flags,
    )


# --------- Convenience wrappers for CausalKit outputs ---------

ResultLike = Union[Dict[str, Any], Any]


def _extract_diag_from_result(res: ResultLike) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Extract m_hat, D, and trimming epsilon from dml_ate/dml_att result dict or model.
    Accepts:
    - dict returned by dml_ate/dml_att (prefers key 'diagnostic_data'; otherwise uses 'model'), or
    - a fitted IRM/DoubleMLIRM-like model instance with a .data attribute.
    Returns (m_hat, D, trimming_threshold_if_any).
    """

    def _try_irm_like(model: Any) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[float]]]:
        """Attempt to extract from our internal IRM (has m_hat_ and data.get_df())."""
        try:
            m_hat = np.asarray(model.m_hat_, dtype=float)
            df = model.data.get_df()
            d = df[model.data.treatment.name].to_numpy().astype(int)
            thr = getattr(model, "trimming_threshold", None)
            thr = float(thr) if thr is not None else None
            return m_hat, d, thr
        except Exception:
            return None

    def _try_doubleml_like(model: Any) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[float]]]:
        """Attempt to extract from a DoubleMLIRM-like object (model.data is DoubleMLData)."""
        try:
            data_obj = getattr(model, "data", None)
            if data_obj is None:
                return None
            # DoubleMLData exposes .data (DataFrame) and lists of columns
            df = getattr(data_obj, "data", None)
            d_cols = getattr(data_obj, "d_cols", None)
            x_cols = getattr(data_obj, "x_cols", None)
            if df is None or d_cols is None or x_cols is None:
                return None
            tname = d_cols[0] if isinstance(d_cols, (list, tuple)) else d_cols
            d = df[tname].to_numpy().astype(int)
            n = d.shape[0]

            # Try to find propensity-like attribute on the model first
            cand_names = [
                "m_hat_", "m_hat", "p_hat_", "p_hat", "propensity_", "propensity", "ps_hat_", "ps_hat", "m",
            ]
            for nm in cand_names:
                val = getattr(model, nm, None)
                if val is not None:
                    arr = np.asarray(val, dtype=float).reshape(-1)
                    if arr.shape[0] == n:
                        thr = getattr(model, "trimming_threshold", None)
                        thr = float(thr) if thr is not None else None
                        return arr, d, thr

            # No reliable cross-fitted propensity found on the model; do not fit in-sample fallback.
            return None
        except Exception:
            return None

    # Dict result path
    if isinstance(res, dict):
        if "diagnostic_data" in res and isinstance(res["diagnostic_data"], dict):
            dd = res["diagnostic_data"]
            m = dd.get("m_hat", None)
            d = dd.get("d", None)
            if m is not None and d is not None:
                m_hat = np.asarray(m, dtype=float)
                d = np.asarray(d, dtype=int)
                thr = dd.get("trimming_threshold", None)
                thr = float(thr) if isinstance(thr, (int, float)) else None
                return m_hat, d, thr
        if "model" in res:
            model = res["model"]
            # Try IRM-like
            out = _try_irm_like(model)
            if out is not None:
                return out
            # Try DoubleML-like
            out = _try_doubleml_like(model)
            if out is not None:
                return out
            raise ValueError("Unsupported result['model'] type; expected IRM-like or DoubleML-like.")
        raise ValueError("Result dict must contain 'diagnostic_data' or 'model'.")

    # Model instance path (IRM or DoubleMLIRM)
    model = res
    out = _try_irm_like(model)
    if out is not None:
        return out
    out = _try_doubleml_like(model)
    if out is not None:
        return out
    raise ValueError("Unsupported result type; pass dml_ate/dml_att result dict or IRM/DoubleMLIRM instance.")


def overlap_report_from_result(
    res: ResultLike,
    *,
    use_hajek: bool = False,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    n_bins: int = 10,
    cal_thresholds: Optional[Dict[str, float]] = None,
    auc_flip_margin: float = 0.05,
) -> Dict[str, Any]:
    """High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.

    If the input result contains a flag indicating normalized IPW (Hájek), this function will
    auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
    dml_ate(normalize_ipw=True) get meaningful ipw_sum_* checks without extra arguments.
    """
    # Auto-detect Hájek/normalized IPW from result where possible
    detected_hajek = bool(use_hajek)
    try:
        if isinstance(res, dict):
            if "diagnostic_data" in res and isinstance(res["diagnostic_data"], dict):
                dd = res["diagnostic_data"]
                if "normalize_ipw" in dd:
                    detected_hajek = bool(dd["normalize_ipw"]) or detected_hajek
            if "model" in res and hasattr(res["model"], "normalize_ipw"):
                detected_hajek = bool(getattr(res["model"], "normalize_ipw")) or detected_hajek
        else:
            # Model instance path
            if hasattr(res, "normalize_ipw"):
                detected_hajek = bool(getattr(res, "normalize_ipw")) or detected_hajek
    except Exception:
        # be permissive; fall back to the explicit flag
        detected_hajek = bool(use_hajek)

    m_hat, d, thr = _extract_diag_from_result(res)
    m_clip = (thr, 1.0 - thr) if thr is not None else None
    return positivity_overlap_checks(
        m_hat=m_hat,
        D=d,
        m_clipped_from=m_clip,
        g_clipped_share=None,
        use_hajek=detected_hajek,
        thresholds=thresholds,
        n_bins=int(n_bins),
        cal_thresholds=cal_thresholds,
        auc_flip_margin=float(auc_flip_margin),
    )


def run_overlap_diagnostics(
    res: ResultLike = None,
    *,
    m_hat: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    n_bins: int = 10,
    use_hajek: Optional[bool] = None,
    m_clipped_from: Optional[Tuple[float, float]] = None,
    g_clipped_share: Optional[float] = None,
    return_summary: bool = True,
    cal_thresholds: Optional[Dict[str, float]] = None,
    auc_flip_margin: float = 0.05,
) -> Dict[str, Any]:
    """
    Single entry-point for overlap / positivity / calibration diagnostics.

    You can call it in TWO ways:
      A) With raw arrays:
         run_overlap_diagnostics(m_hat=..., D=...)
      B) With a model/result:
         run_overlap_diagnostics(res=<dml_ate/dml_att result dict or IRM/DoubleML-like model>)

    The function:
      - Auto-extracts (m_hat, D, trimming_threshold) from `res` if provided.
      - Auto-detects Hájek normalization if available on `res` (normalize_ipw).
      - Runs positivity/overlap checks (edge mass, KS, AUC, ESS, tails, ATT identity),
        clipping audit, and calibration (ECE + logistic recalibration).
      - Returns a dict with full details and, optionally, a compact summary DataFrame.

    Returns
    -------
    dict with keys including:
      - n, n_treated, p1
      - edge_mass, edge_mass_by_arm, ks, auc
      - ate_ipw, ate_ess, ate_tails
      - att_weights, att_ess
      - clipping
      - calibration (with reliability_table)
      - flags  (GREEN/YELLOW/RED/NA)
      - summary (pd.DataFrame) if return_summary=True
      - meta (use_hajek, thresholds)
    """
    # ---- Resolve inputs (arrays vs result/model) ----
    if m_hat is None or D is None:
        if res is None:
            raise ValueError("Pass either (m_hat, D) or `res`.")
        m_hat_res, d_res, thr = _extract_diag_from_result(res)
        m_hat, D = m_hat_res, d_res
        # Inherit trimming as clipping audit bounds if not specified
        if m_clipped_from is None and thr is not None:
            m_clipped_from = (thr, 1.0 - thr)

        # Auto-detect Hájek if not specified
        detected_hajek = False
        try:
            if isinstance(res, dict):
                dd = res.get("diagnostic_data", {})
                detected_hajek = bool(dd.get("normalize_ipw", False))
                if not detected_hajek and "model" in res and hasattr(res["model"], "normalize_ipw"):
                    detected_hajek = bool(getattr(res["model"], "normalize_ipw"))
            else:
                if hasattr(res, "normalize_ipw"):
                    detected_hajek = bool(getattr(res, "normalize_ipw"))
        except Exception:
            detected_hajek = False
        if use_hajek is None:
            use_hajek = detected_hajek
    else:
        if use_hajek is None:
            use_hajek = False

    # ---- Run main checks (uses your existing implementation) ----
    report = positivity_overlap_checks(
        m_hat=m_hat,
        D=D,
        m_clipped_from=m_clipped_from,
        g_clipped_share=g_clipped_share,
        use_hajek=use_hajek,
        thresholds=thresholds,
        n_bins=int(n_bins),
        cal_thresholds=cal_thresholds,
        auc_flip_margin=float(auc_flip_margin),
    )

    # ---- Build a compact summary table (optional) ----
    if return_summary:
        f = report["flags"]
        em = report["edge_mass"]
        tails_w1 = report["ate_tails"]["w1"]
        tails_w0 = report["ate_tails"]["w0"]
        clip = report["clipping"]
        cal = report["calibration"] or {}
        cal_ece = cal.get("ece", np.nan)
        cal_slope = (cal.get("recalibration") or {}).get("slope", np.nan)
        cal_int = (cal.get("recalibration") or {}).get("intercept", np.nan)
        clip_total = (clip.get("m_clip_lower", np.nan) + clip.get("m_clip_upper", np.nan)
                      if not (np.isnan(clip.get("m_clip_lower", np.nan)) or np.isnan(clip.get("m_clip_upper", np.nan)))
                      else np.nan)

        def _safe_ratio(num, den):
            try:
                num_f = float(num)
                den_f = float(den)
            except Exception:
                return np.nan
            if not (np.isfinite(num_f) and np.isfinite(den_f)) or den_f == 0.0:
                return np.nan
            return num_f / den_f

        summary_rows = [
            {"metric": "edge_0.01_below",      "value": em["share_below_001"],                 "flag": f["edge_mass_001"]},
            {"metric": "edge_0.01_above",      "value": em["share_above_001"],                 "flag": f["edge_mass_001"]},
            {"metric": "edge_0.02_below",      "value": em["share_below_002"],                 "flag": f["edge_mass_002"]},
            {"metric": "edge_0.02_above",      "value": em["share_above_002"],                 "flag": f["edge_mass_002"]},
            {"metric": "KS",                   "value": report["ks"],                           "flag": f["ks"]},
            {"metric": "AUC",                  "value": report["auc"],                          "flag": f["auc"]},
            {"metric": "ESS_treated_ratio",    "value": report["ate_ess"]["ess_ratio_w1"],      "flag": f["ess_w1"]},
            {"metric": "ESS_control_ratio",    "value": report["ate_ess"]["ess_ratio_w0"],      "flag": f["ess_w0"]},
            {"metric": "tails_w1_q99/med",     "value": _safe_ratio(tails_w1.get("q99", np.nan), tails_w1.get("median", np.nan)), "flag": f["tails_w1"]},
            {"metric": "tails_w0_q99/med",     "value": _safe_ratio(tails_w0.get("q99", np.nan), tails_w0.get("median", np.nan)), "flag": f["tails_w0"]},
            {"metric": "ATT_identity_relerr",  "value": report["att_weights"]["rel_err"],       "flag": f["att_identity"]},
            {"metric": "clip_m_total",         "value": clip_total,                             "flag": f["clip_m"]},
            {"metric": "calib_ECE",            "value": cal_ece,                                "flag": f.get("calibration_ece", "NA")},
            {"metric": "calib_slope",          "value": cal_slope,                              "flag": f.get("calibration_slope", "NA")},
            {"metric": "calib_intercept",      "value": cal_int,                                "flag": f.get("calibration_intercept", "NA")},
        ]
        report["summary"] = pd.DataFrame(summary_rows)

    report["meta"] = dict(use_hajek=bool(use_hajek), thresholds=thresholds, n_bins=int(n_bins))
    return report


def att_overlap_tests(dml_att_result: dict, epsilon_list=(0.01, 0.02)) -> dict:
    """
    Compute ATT overlap/weight diagnostics from a dml_att(_source) result dict.

    Inputs expected in result['diagnostic_data']:
      - m_hat: np.ndarray of cross-fitted propensity scores Pr(D=1|X)
      - d: np.ndarray of treatment indicators {0,1}

    Returns:
      dict with keys:
        - edge_mass: {'eps': {eps: {'share_below': float, 'share_above': float, 'warn': bool}}}
        - ks: {'value': float, 'warn': bool}
        - auc: {'value': float or nan, 'flag': str}  # 'GREEN'/'YELLOW'/'RED' or 'NA' if undefined
        - ess: {'treated': {'ess': float, 'n': int, 'ratio': float, 'flag': str},
                'control': {'ess': float, 'n': int, 'ratio': float, 'flag': str}}
        - att_weight_identity: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float, 'flag': str}
    """
    dd = dml_att_result.get("diagnostic_data", {})
    m = np.asarray(dd.get("m_hat", None), dtype=float)
    D = np.asarray(dd.get("d", None))
    if m is None or D is None or m.size == 0:
        raise ValueError("diagnostic_data with 'm_hat' and 'd' is required")

    n = len(m)
    if len(D) != n:
        raise ValueError("Length mismatch between m_hat and d")

    # 1) Edge mass
    edge = {}
    for eps in epsilon_list:
        share_below = float(np.mean(m < eps))
        share_above = float(np.mean(m > 1.0 - eps))
        warn = (share_below > 0.02) or (share_above > 0.02)
        if np.isclose(eps, 0.02):
            warn = (share_below > 0.05) or (share_above > 0.05)
        edge.setdefault("eps", {})[eps] = {
            "share_below": share_below,
            "share_above": share_above,
            "warn": warn,
        }

    # 2) KS distance between m|D=1 and m|D=0
    m1 = m[D == 1]
    m0 = m[D == 0]
    ks_val = float(_ks_statistic(m1, m0)) if (len(m1) > 1 and len(m0) > 1) else float("nan")
    ks_warn = (ks_val > 0.25) if not np.isnan(ks_val) else False

    # 3) AUC of m vs D (OOS not available: uses given m_hat)
    auc_flag = "NA"
    auc_val = float("nan")
    flip_hint = None
    if len(np.unique(D)) == 2 and np.all((m >= 0) & (m <= 1)):
        try:
            auc_val = float(_auc_mann_whitney(m, D.astype(int)))
            sep = max(auc_val, 1.0 - auc_val)
            if sep > 0.95:
                auc_flag = "RED"
            elif sep > 0.90:
                auc_flag = "YELLOW"
            else:
                auc_flag = "GREEN"
            if auc_val < 0.45:
                flip_hint = "YELLOW"
        except Exception:
            auc_flag = "NA"

    # 4) ESS for ATE-style weights (requested)
    #    w1 = D/m, w0=(1-D)/(1-m)
    #    For ESS we consider group-specific non-zero weights.
    def ess_ratio(w, mask):
        wg = w[mask]
        if wg.size == 0:
            return 0.0, 0, 0.0, "NA"
        s = float(np.sum(wg))
        s2 = float(np.sum(wg ** 2))
        ess = (s * s) / s2 if s2 > 0 else 0.0
        n_g = int(np.sum(mask))
        ratio = ess / max(n_g, 1)
        flag = "GREEN"
        if ratio < 0.15:
            flag = "RED"
        elif ratio < 0.30:
            flag = "YELLOW"
        return ess, n_g, ratio, flag

    # Avoid division by zero by clipping m to (eps,1-eps)
    m_safe = np.clip(m, 1e-8, 1 - 1e-8)
    w1 = D.astype(float) / m_safe
    w0 = (1 - D.astype(float)) / (1 - m_safe)

    ess1, n1, r1, f1 = ess_ratio(w1, D == 1)
    ess0, n0, r0, f0 = ess_ratio(w0, D == 0)

    # 5) ATT weight-sum identity:
    #    sum (1-D) * m/(1-m)  ?=  sum D
    lhs = float(np.sum((1 - D.astype(float)) * (m_safe / (1 - m_safe))))
    rhs = float(np.sum(D))
    rel_err = abs(lhs - rhs) / rhs if rhs > 0 else float("inf")
    att_flag = "GREEN" if rel_err <= 0.05 else "YELLOW"

    return {
        "edge_mass": edge,
        "ks": {"value": ks_val, "warn": ks_warn},
        "auc": {"value": auc_val, "flag": auc_flag, "flip_suspected": (flip_hint or "NA")},
        "ess": {
            "treated": {"ess": ess1, "n": n1, "ratio": r1, "flag": f1},
            "control": {"ess": ess0, "n": n0, "ratio": r0, "flag": f0},
        },
        "att_weight_identity": {"lhs_sum": lhs, "rhs_sum": rhs, "rel_err": rel_err, "flag": att_flag},
    }
