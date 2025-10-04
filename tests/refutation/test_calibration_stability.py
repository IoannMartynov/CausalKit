import numpy as np
import warnings

from causalkit.refutation.overlap.overlap_validation import calibration_report_m


def test_calibration_no_runtime_warnings_extreme_inputs():
    # Construct nearly separable data with extreme probabilities
    n = 1000
    p = np.zeros(n, dtype=float)
    p[n//2:] = 1.0
    y = np.zeros(n, dtype=int)
    y[n//2:] = 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=RuntimeWarning)
        rep = calibration_report_m(p, y, n_bins=10)

    # No RuntimeWarnings should be raised by calibration
    runtime_warns = [wt for wt in w if issubclass(wt.category, RuntimeWarning)]
    assert len(runtime_warns) == 0

    # Recalibration parameters should be finite numbers
    rc = rep["recalibration"]
    assert np.isfinite(rc["intercept"]) and np.isfinite(rc["slope"])