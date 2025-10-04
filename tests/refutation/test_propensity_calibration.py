import numpy as np
import pandas as pd

from causalkit.refutation.overlap.overlap_validation import (
    ece_binary,
    calibration_report_m,
)


def test_ece_binary_basic():
    # Perfectly calibrated edge cases
    p = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0, 1, 0, 1])
    ece = ece_binary(p, y, n_bins=2)
    assert np.isfinite(ece)
    assert abs(ece - 0.0) < 1e-12


def _simulate_probs_and_labels(n=4000, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    # true probability
    p = 1.0 / (1.0 + np.exp(-(0.8 * x)))
    y = rng.binomial(1, p)
    return p, y


def test_calibration_report_keys_and_flags_green_on_well_calibrated():
    p, y = _simulate_probs_and_labels(n=3000, seed=11)
    rep = calibration_report_m(p, y, n_bins=10)

    # Basic keys present
    assert isinstance(rep, dict)
    for k in ["auc", "brier", "ece", "reliability_table", "recalibration", "flags"]:
        assert k in rep

    # Reliability table shape
    rel = rep["reliability_table"]
    assert hasattr(rel, "shape")
    assert rel.shape[1] >= 5

    # Well-calibrated model should not be flagged RED
    assert rep["flags"]["ece"] in {"GREEN", "YELLOW"}
    assert rep["flags"]["slope"] in {"GREEN", "YELLOW"}
    assert rep["flags"]["intercept"] in {"GREEN", "YELLOW"}


def test_misscaled_scores_trigger_flags():
    p, y = _simulate_probs_and_labels(n=4000, seed=21)
    # mis-scale probabilities to create calibration slope far from 1
    z = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    p_bad = 1.0 / (1.0 + np.exp(-0.5 * z))  # slope ~ 0.5 expected

    rep = calibration_report_m(p_bad, y, n_bins=10)

    # Expect strong slope flag
    assert rep["flags"]["slope"] == "RED"
    # ECE should be finite and likely above warn
    assert np.isfinite(rep["ece"]) and rep["ece"] >= 0.0
