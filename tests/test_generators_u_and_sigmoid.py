import numpy as np
import pandas as pd

from causalkit.data.generators import CausalDatasetGenerator, _sigmoid


def test_sigmoid_supports_ndarrays_and_bounds():
    z = np.array([
        [-1e6, -100.0, -10.0, 0.0, 10.0, 100.0, 1e6],
        [-50.0, -1.0, 0.0, 1.0, 50.0, 1000.0, -1000.0],
    ], dtype=float)
    s = _sigmoid(z)
    assert s.shape == z.shape
    assert np.all(np.isfinite(s))
    assert np.all((s > 0.0) & (s < 1.0))
    # Monotone in each row where z is increasing
    assert np.all(np.diff(s[0]) > 0)


def test_g_columns_marginalize_over_u_for_binary_when_u_strength_y_nonzero():
    # No X, no treatment effect; only intercept and U -> g0 == g1 constant across rows
    gen = CausalDatasetGenerator(
        theta=0.0,
        beta_y=None,
        beta_d=None,
        alpha_y=0.3,
        alpha_d=0.0,
        sigma_y=1.0,
        outcome_type="binary",
        k=0,
        u_strength_d=0.0,
        u_strength_y=1.0,
        seed=123,
    )
    n = 5000
    df = gen.generate(n)

    # g0 and g1 should be constant across rows (no X), hence zero variance
    assert np.allclose(df["g0"].to_numpy(), df["g0"].iloc[0])
    assert np.allclose(df["g1"].to_numpy(), df["g1"].iloc[0])

    # oracle_nuisance should agree with the dataset's g0/g1
    m_fn, g0_fn, g1_fn = gen.oracle_nuisance()
    g0_theory = g0_fn(np.empty((0,), dtype=float))
    g1_theory = g1_fn(np.empty((0,), dtype=float))
    assert abs(df["g0"].mean() - g0_theory) < 1e-6
    assert abs(df["g1"].mean() - g1_theory) < 1e-6
