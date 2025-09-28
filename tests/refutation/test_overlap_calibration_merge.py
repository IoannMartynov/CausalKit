import numpy as np

from causalkit.refutation.overlap.overlap_validation import positivity_overlap_checks


def test_positivity_overlap_checks_includes_calibration():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-x))
    d = rng.binomial(1, p)

    report = positivity_overlap_checks(p, d)

    # calibration sub-report present
    assert "calibration" in report
    cal = report["calibration"]
    assert isinstance(cal, dict)
    assert set(["auc", "brier", "ece", "reliability_table", "recalibration", "flags"]).issubset(cal.keys())

    # merged flags present and valid labels
    flags = report.get("flags", {})
    for key in ("calibration_ece", "calibration_slope", "calibration_intercept"):
        assert key in flags
        assert flags[key] in {"GREEN", "YELLOW", "RED", "NA"}
