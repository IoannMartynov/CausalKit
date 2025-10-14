import numpy as np
import pytest

from causalis.data.generators import CausalDatasetGenerator


def test_poisson_overflow_guard_clips_link_scale():
    # Extremely large alpha_y would overflow exp without clipping
    gen = CausalDatasetGenerator(outcome_type="poisson", alpha_y=1000.0, k=0, seed=0)
    df = gen.generate(256)

    # y, mu0, mu1 should be finite and mu0/mu1 bounded by exp(20)
    assert np.isfinite(df["y"]).all()
    assert np.isfinite(df["mu0"]).all()
    assert np.isfinite(df["mu1"]).all()
    max_allowed = np.exp(20.0) + 1e-6
    assert float(df["mu0"].max()) <= max_allowed
    assert float(df["mu1"].max()) <= max_allowed


def test_calibrator_bracket_fallback_clamps_to_endpoint():
    # Force treatment scores so large that even alpha_t in [-50,50] can't bracket target
    confounder_specs = [{"name": "x", "dist": "normal", "mu": 100.0, "sd": 0.0}]
    gen = CausalDatasetGenerator(
        confounder_specs=confounder_specs,
        beta_d=np.array([100.0], dtype=float),
        target_d_rate=0.2,
        outcome_type="continuous",
        seed=123,
    )

    df = gen.generate(2000)

    # alpha_t should be clamped very close to an endpoint (-50 or 50)
    assert (gen.alpha_d > 49.0) or (gen.alpha_d < -49.0)
    # Realized treatment rate should be saturated near 0 or 1 (here near 1)
    t_rate = float(df["d"].mean())
    assert (t_rate > 0.95) or (t_rate < 0.05)
