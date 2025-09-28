import re
import numpy as np

from causalkit.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_analysis


class DummyModel:
    def _sensitivity_element_est(self):
        # Provide simple, finite elements
        elems = {
            "sigma2": 1.0,
            "psi_sigma2": np.zeros(2),
            # simple m_alpha and rr vectors
            "m_alpha": np.array([0.5, 0.5]),
            "riesz_rep": np.array([2.0, -1.0]),
        }
        return elems


def _extract_bounds(summary_text: str):
    # parse the line with CI and theta bounds
    for line in summary_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "theta" and len(parts) >= 6:
            # Expected columns: name, CI lower, theta lower, theta, theta upper, CI upper
            ci_low = float(parts[1])
            th_low = float(parts[2])
            th = float(parts[3])
            th_up = float(parts[4])
            ci_up = float(parts[5])
            return ci_low, th_low, th, th_up, ci_up
    raise AssertionError("Failed to parse theta row from summary")


def test_top_level_sensitivity_analysis_uses_local_helpers_and_rho_sign():
    model = DummyModel()
    # Provide coefficient, se, and CI so the fallback path engages without needing model.confint
    effect = {
        "model": model,
        "coefficient": 1.0,
        "std_error": 0.2,
        "confidence_interval": (0.6, 1.4),
    }

    # Positive rho should produce wider bias-based bounds than negative rho (same |rho|)
    pos = sensitivity_analysis(effect, cf_y=0.1, cf_d=0.2, rho=0.9, level=0.95)
    neg = sensitivity_analysis(effect, cf_y=0.1, cf_d=0.2, rho=-0.9, level=0.95)

    # Check header includes SNR proxy label
    assert "Robustness (SNR proxy)" in pos

    _, th_low_pos, th_pos, th_up_pos, _ = _extract_bounds(pos)
    _, th_low_neg, th_neg, th_up_neg, _ = _extract_bounds(neg)

    width_pos = th_up_pos - th_pos
    width_neg = th_up_neg - th_neg

    assert width_neg <= width_pos + 1e-9
    # symmetric around theta
    assert abs((th_pos - th_low_pos) - (th_up_pos - th_pos)) < 1e-8
    assert abs((th_neg - th_low_neg) - (th_up_neg - th_neg)) < 1e-8
