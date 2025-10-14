import numpy as np
import math
import pytest

from causalis.refutation.overlap.overlap_validation import (
    _ks_statistic,
    ks_distance,
    positivity_overlap_checks,
    att_overlap_tests,
)


def test_ks_ties_identical_samples_zero_distance():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    d = _ks_statistic(a, b)
    assert d == 0.0


def test_ks_distance_identical_groups_zero():
    # Construct identical within-group distributions with ties
    m = np.array([0.0, 0.0, 0.0, 0.0])
    D = np.array([1, 1, 0, 0])
    d = ks_distance(m, D)
    assert d == 0.0


def test_clip_flag_is_na_when_unknown():
    rng = np.random.default_rng(0)
    m = rng.uniform(0.1, 0.9, size=200)
    D = rng.integers(0, 2, size=200)
    rep = positivity_overlap_checks(m, D, m_clipped_from=None)
    assert rep["flags"].get("clip_m") == "NA"


def test_att_auc_separation_and_flip_hint():
    # Create anti-predictive scores: high m for control, low for treated
    n = 200
    D = np.array([0, 1] * (n // 2))
    rng = np.random.default_rng(1)
    # m ≈ 1 - D plus tiny noise, clipped to [0,1]
    m = np.clip(1 - D + rng.normal(0, 0.01, size=n), 0.0, 1.0)
    res = {"diagnostic_data": {"m_hat": m, "d": D}}
    out = att_overlap_tests(res)
    auc = out["auc"]["value"]
    flag = out["auc"]["flag"]
    flip = out["auc"].get("flip_suspected")
    assert not math.isnan(auc)
    # Separation should be extreme, so RED
    assert flag == "RED"
    # Flip suspected should be YELLOW since auc < 0.45
    assert flip == "YELLOW"