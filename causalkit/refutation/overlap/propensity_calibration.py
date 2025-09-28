from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# We reuse the AUC implementation from the overlap diagnostics to avoid extra deps
from .overlap_validation import _auc_mann_whitney


# Thresholds for flags (kept local to this module)
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
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_recalibration(p: np.ndarray, y: np.ndarray, *, max_iter: int = 100, tol: float = 1e-8, ridge: float = 1e-8) -> tuple[float, float]:
    """
    Fit logistic recalibration model: Pr(D=1|p) = sigmoid(alpha + beta * logit(p)).
    Uses IRLS/Newton steps with a tiny ridge for numerical stability.

    Returns (alpha, beta).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    z = _logit(p)
    X = np.column_stack([np.ones_like(z), z])  # [1, z]
    theta = np.array([np.log(np.mean(y) / (1 - np.mean(y) + 1e-12) + 1e-12), 1.0], dtype=float)

    for _ in range(max_iter):
        eta = X @ theta
        mu = _sigmoid(eta)
        # Guard against degenerate probabilities
        mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
        # Gradient and Hessian
        r = y - mu  # residuals
        W = mu * (1.0 - mu)
        # Build X^T W X and X^T r
        XT_W = X.T * W  # broadcasting
        H = XT_W @ X
        # Ridge for stability
        H[0, 0] += ridge
        H[1, 1] += ridge
        g = X.T @ r
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            step = np.linalg.pinv(H) @ g
        theta_new = theta + step
        if np.max(np.abs(step)) < tol:
            theta = theta_new
            break
        theta = theta_new

    alpha, beta = float(theta[0]), float(theta[1])
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


__all__ = ["ece_binary", "calibration_report_m", "CAL_THRESHOLDS"]
