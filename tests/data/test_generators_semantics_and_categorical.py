import numpy as np
import pandas as pd

from causalis.data.generators import CausalDatasetGenerator


def test_continuous_g_excludes_U_when_u_strength_y_nonzero():
    # Continuous outcomes: g0/g1 should exclude U (mean-zero unobserved)
    gen = CausalDatasetGenerator(
        theta=0.7,
        alpha_y=0.1,
        alpha_d=0.0,
        outcome_type="continuous",
        sigma_y=1.0,
        k=3,
        seed=123,
        u_strength_y=1.5,
        beta_y=np.array([0.2, -0.1, 0.3]),
    )
    n = 1000
    df = gen.generate(n)

    # Recompute oracle using generator's oracle_nuisance
    m_fn, g0_fn, g1_fn = gen.oracle_nuisance()
    X = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    g0_row = np.array([g0_fn(x) for x in X])
    g1_row = np.array([g1_fn(x) for x in X])

    assert np.allclose(df["g0"].to_numpy(), g0_row, atol=1e-3)
    assert np.allclose(df["g1"].to_numpy(), g1_row, atol=1e-3)


def test_m_and_mobs_semantics():
    # When u_strength_d != 0, m (marginal) and m_obs (realized) should generally differ per-row
    gen = CausalDatasetGenerator(
        theta=0.0,
        outcome_type="binary",
        sigma_y=1.0,
        u_strength_d=1.2,
        k=2,
        seed=42,
    )
    df = gen.generate(8000)

    assert "m" in df.columns and "m_obs" in df.columns

    mobs_mean = float(df["m_obs"].mean())
    d_rate = float(df["d"].mean())
    # Realized propensity mean should match empirical treatment rate
    assert abs(mobs_mean - d_rate) < 2e-2

    # The arrays should not be (near-)identical when u affects treatment
    assert not np.allclose(df["m"].to_numpy(), df["m_obs"].to_numpy(), atol=1e-3)


def test_categorical_probs_normalized_non_copula_and_copula():
    # Non-copula categorical with unnormalized probs should work and one-hot sum <= n
    gen_nc = CausalDatasetGenerator(
        theta=0.0,
        outcome_type="continuous",
        sigma_y=1.0,
        seed=1,
        confounder_specs=[{"name": "cat", "dist": "categorical", "categories": [0, 1, 2], "probs": [2, 3, 5]}],
    )
    df_nc = gen_nc.generate(1000)
    cols_nc = [c for c in df_nc.columns if c.startswith("cat_")]
    # one-hot for levels 1 and 2
    assert len(cols_nc) == 2
    # one-hot columns should not exceed n when summed
    assert df_nc[cols_nc].to_numpy().sum() <= 1000 + 1e-6

    # Copula categorical path: ensure no index errors at boundaries and normalization is applied
    gen_c = CausalDatasetGenerator(
        theta=0.0,
        outcome_type="continuous",
        sigma_y=1.0,
        seed=2,
        use_copula=True,
        confounder_specs=[{"name": "catc", "dist": "categorical", "categories": [0, 1, 2], "probs": [2, 3, 5]}],
    )
    df_c = gen_c.generate(2000)
    cols_c = [c for c in df_c.columns if c.startswith("catc_")]
    assert len(cols_c) == 2
    assert df_c[cols_c].to_numpy().sum() <= 2000 + 1e-6
