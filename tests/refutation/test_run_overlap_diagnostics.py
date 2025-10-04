import numpy as np
import pandas as pd

from causalkit.refutation.overlap.overlap_validation import run_overlap_diagnostics


def test_run_overlap_diagnostics_with_arrays_returns_summary_and_meta():
    rng = np.random.default_rng(42)
    n = 300
    # moderately overlapping propensities
    m = np.clip(rng.beta(2, 2, size=n), 1e-6, 1-1e-6)
    D = rng.integers(0, 2, size=n)

    rep = run_overlap_diagnostics(m_hat=m, D=D)

    assert isinstance(rep, dict)
    assert rep.get("n") == n
    assert "flags" in rep and isinstance(rep["flags"], dict)

    # Summary DataFrame exists by default
    assert "summary" in rep
    assert isinstance(rep["summary"], pd.DataFrame)
    # Check a couple of expected metrics exist
    mets = set(rep["summary"]["metric"].tolist())
    assert {"KS", "AUC", "ESS_treated_ratio", "ESS_control_ratio"}.issubset(mets)

    # Meta includes use_hajek (default False here) and thresholds
    assert "meta" in rep and isinstance(rep["meta"], dict)
    assert rep["meta"].get("use_hajek") in (False, True)
    assert isinstance(rep["meta"].get("thresholds"), dict)


def test_run_overlap_diagnostics_with_result_autodetects_hajek_and_clipping():
    rng = np.random.default_rng(0)
    n = 500
    m = rng.uniform(0.0, 1.0, size=n)
    D = rng.integers(0, 2, size=n)

    res = {
        "diagnostic_data": {
            "m_hat": m,
            "d": D,
            "trimming_threshold": 0.10,
            "normalize_ipw": True,
        }
    }

    rep = run_overlap_diagnostics(res=res)

    # Hájek should be auto-detected from the result
    assert rep["meta"].get("use_hajek") is True

    # Clipping audit should be populated (not NA since thr provided)
    clip = rep.get("clipping", {})
    assert np.isfinite(clip.get("m_clip_lower", np.nan))
    assert np.isfinite(clip.get("m_clip_upper", np.nan))

    # Flags should include clip_m not marked as NA
    assert rep["flags"].get("clip_m") in {"GREEN", "YELLOW", "RED"}

    # Summary exists
    assert isinstance(rep.get("summary"), pd.DataFrame)
