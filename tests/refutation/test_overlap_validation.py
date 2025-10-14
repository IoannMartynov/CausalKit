import math
import numpy as np
import pytest

from causalis.refutation.overlap.overlap_validation import (
    auc_for_m,
    positivity_overlap_checks,
    att_weight_sum_identity,
)


def test_auc_ties_correct():
    # Scores with ties and mixed labels
    scores = np.array([0.5, 0.5, 0.8, 0.3], dtype=float)
    labels = np.array([1, 0, 1, 0], dtype=int)
    # Expected AUC: pairs = 4; favorable = 3.5 (including 0.5 for the tie) => 0.875
    auc = auc_for_m(scores, labels)
    assert pytest.approx(auc, rel=1e-12) == 0.875


def test_ate_weights_clipping_hajek_means_are_one_and_finite():
    # Include extreme propensities that would cause inf without clipping
    m_hat = np.array([0.0, 1.0, 0.5, 1e-16, 1 - 1e-16], dtype=float)
    D = np.array([1, 0, 1, 1, 0], dtype=int)

    rep = positivity_overlap_checks(m_hat, D, use_hajek=True)
    ate = rep["ate_ipw"]

    # Check finiteness of sums and means
    assert math.isfinite(ate["sum_w1"]) and math.isfinite(ate["sum_w0"])  # no inf/NaN after clipping

    # Under Hájek normalization, group means should be ~1 where defined
    if not math.isnan(ate["mean_w1"]):
        assert pytest.approx(ate["mean_w1"], rel=1e-9, abs=1e-9) == 1.0
    if not math.isnan(ate["mean_w0"]):
        assert pytest.approx(ate["mean_w0"], rel=1e-9, abs=1e-9) == 1.0


def test_att_identity_unified_relerr_and_flag():
    rng = np.random.default_rng(0)
    n = 200
    # Random propensities away from exact 0/1 to avoid trivialities
    m_hat = rng.uniform(0.02, 0.98, size=n)
    D = rng.binomial(1, 0.4, size=n)

    # Ground truth via helper
    att_id = att_weight_sum_identity(m_hat, D)

    rep = positivity_overlap_checks(m_hat, D)

    # Returned att_weights should reflect the helper's numbers
    aw = rep["att_weights"]
    assert pytest.approx(aw["lhs_sum"], rel=1e-12) == att_id["lhs_sum"]
    assert pytest.approx(aw["rhs_sum"], rel=1e-12) == att_id["rhs_sum"]
    assert pytest.approx(aw["rel_err"], rel=1e-12) == att_id["rel_err"]

    # Flag agrees with thresholds
    thr = rep["flags"]
    re = aw["rel_err"]
    if math.isnan(re):
        assert thr["att_identity"] == "NA"
    elif re > rep["flags"].get("ipw_relerr_strong", 0) if False else False:
        # not used; thresholds are in defaults; verify against DEFAULT_THRESHOLDS instead
        pass
    else:
        # Retrieve thresholds used in the function
        # They are not returned explicitly; re-evaluate the logic here with known defaults
        # DEFAULT: strong=0.10, warn=0.05
        strong = 0.10
        warn = 0.05
        expected = "RED" if re > strong else ("YELLOW" if re > warn else "GREEN")
        assert thr["att_identity"] == expected


def test_ipw_sum_flags_only_under_hajek():
    # Some simple data
    m_hat = np.array([0.2, 0.8, 0.6, 0.4])
    D = np.array([1, 0, 1, 0])

    # When not using Hájek, ipw_sum_* flags should be NA
    rep_no_hajek = positivity_overlap_checks(m_hat, D, use_hajek=False)
    assert rep_no_hajek["flags"]["ipw_sum_w1"] == "NA"
    assert rep_no_hajek["flags"]["ipw_sum_w0"] == "NA"

    # With Hájek, they should be defined and GREEN (by construction)
    rep_hajek = positivity_overlap_checks(m_hat, D, use_hajek=True)
    assert rep_hajek["flags"]["ipw_sum_w1"] in {"GREEN", "YELLOW", "RED", "NA"}  # at least not missing
    assert rep_hajek["flags"]["ipw_sum_w1"] != "NA"
    assert rep_hajek["flags"]["ipw_sum_w0"] != "NA"


def test_overlap_report_autodetects_normalize_ipw_from_result_dict():
    # Craft a minimal dml_ate-like result with diagnostic_data
    m_hat = np.array([0.3, 0.7, 0.6, 0.4], dtype=float)
    d = np.array([1, 0, 1, 0], dtype=int)
    res = {
        "diagnostic_data": {
            "m_hat": m_hat,
            "d": d,
            "normalize_ipw": True,  # should trigger Hájek logic automatically
        }
    }
    from causalis.refutation.overlap.overlap_validation import overlap_report_from_result

    rep = overlap_report_from_result(res)  # no explicit use_hajek passed
    # ipw_sum_* flags should not be NA because normalize_ipw=True was detected
    assert rep["flags"]["ipw_sum_w1"] != "NA"
    assert rep["flags"]["ipw_sum_w0"] != "NA"


def test_auc_flip_guard_and_flag():
    # Construct a flipped-score scenario: lower scores for treated
    m_hat = np.array([0.1, 0.9, 0.1, 0.9], dtype=float)
    D = np.array([1, 0, 1, 0], dtype=int)
    rep = positivity_overlap_checks(m_hat, D)
    # Since auc ~ 0.0, separability is 1.0 -> RED, and flip suspected
    assert rep["flags"]["auc"] == "RED"
    assert rep["flags"].get("auc_flip_suspected") == "YELLOW"


def test_att_ess_ratio_denominators():
    rng = np.random.default_rng(1)
    n = 300
    m_hat = rng.uniform(0.05, 0.95, size=n)
    D = rng.binomial(1, 0.4, size=n)
    rep = positivity_overlap_checks(m_hat, D)
    att = rep["att_ess"]
    n1 = int(D.sum())
    n0 = int((~D.astype(bool)).sum())
    if not math.isnan(att["ess_w1"]) and n1 > 0:
        assert pytest.approx(att["ess_ratio_w1"], rel=1e-12) == att["ess_w1"] / n1
    if not math.isnan(att["ess_w0"]) and n0 > 0:
        assert pytest.approx(att["ess_ratio_w0"], rel=1e-12) == att["ess_w0"] / n0


def test_tails_flag_red_on_extreme_blowups():
    # Many treated with moderate m, one extreme small m -> huge w1 tail
    m_hat = np.array([0.5] * 50 + [1e-4])
    D = np.array([1] * 51)
    rep = positivity_overlap_checks(m_hat, D)
    assert rep["flags"]["tails_w1"] == "RED"
