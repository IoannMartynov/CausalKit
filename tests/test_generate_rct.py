import numpy as np
import pandas as pd
import pytest

from causalkit.data.generators import generate_rct, CausalDatasetGenerator


@pytest.mark.parametrize("n,split", [(5000, 0.5), (8000, 0.3)])
def test_randomization_and_independence(n, split):
    df = generate_rct(n=n, split=split, random_state=123, k=5)
    t_mean = df["t"].mean()
    assert abs(t_mean - split) < 0.02  # binomial tolerance
    # correlations with X should be near zero
    xcols = [c for c in df.columns if c.startswith("x")]
    for c in xcols:
        corr = np.corrcoef(df["t"], df[c])[0, 1]
        assert abs(corr) < 0.05


def test_binary_outcome_means():
    n, split = 10000, 0.4
    params = {"p": {"A": 0.1, "B": 0.2}}
    df = generate_rct(n=n, split=split, random_state=42, target_type="binary", target_params=params)
    m0 = df.loc[df.t == 0, "y"].mean()
    m1 = df.loc[df.t == 1, "y"].mean()
    assert abs(m0 - 0.1) < 0.02
    assert abs(m1 - 0.2) < 0.02


def test_normal_outcome_mean_and_std():
    n = 12000
    params = {"mean": {"A": 0.0, "B": 0.5}, "std": 2.0}
    df = generate_rct(n=n, split=0.5, random_state=7, target_type="normal", target_params=params)
    m0 = df.loc[df.t == 0, "y"].mean()
    m1 = df.loc[df.t == 1, "y"].mean()
    s = df["y"].std()
    assert abs(m0 - 0.0) < 0.05
    assert abs(m1 - 0.5) < 0.05
    assert 1.8 < s < 2.2


def test_poisson_outcome_means():
    n = 15000
    params = {"shape": 2.0, "scale": {"A": 1.5, "B": 2.0}}
    lamA = 2.0 * 1.5
    lamB = 2.0 * 2.0
    df = generate_rct(n=n, split=0.5, random_state=99, target_type="poisson", target_params=params)
    m0 = df.loc[df.t == 0, "y"].mean()
    m1 = df.loc[df.t == 1, "y"].mean()
    assert abs(m0 - lamA) < 0.1
    assert abs(m1 - lamB) < 0.1


def test_schema_and_types_and_propensity():
    n, split = 3000, 0.6
    df = generate_rct(n=n, split=split, random_state=11, k=2, add_ancillary=False)
    # Required columns
    required = {"y", "t", "propensity", "mu0", "mu1", "cate"}
    assert required.issubset(df.columns)
    # t in {0,1}
    assert set(df["t"].unique()).issubset({0.0, 1.0})
    # propensity constant and equals split
    assert np.allclose(df["propensity"].to_numpy(), split)


def test_confounder_parity_with_class():
    n = 4000
    seed = 202
    specs = [
        {"name": "z1", "dist": "normal", "mu": 3.0, "sd": 0.5},
        {"name": "z2", "dist": "bernoulli", "p": 0.3},
        {"name": "cat", "dist": "categorical", "categories": ["A", "B", "C"], "probs": [0.2, 0.5, 0.3]},
    ]
    df = generate_rct(n=n, split=0.5, random_state=seed, confounder_specs=specs, add_ancillary=False)

    # Sample X from class with the same seed/specs
    gen = CausalDatasetGenerator(confounder_specs=specs, seed=seed)
    X_class, names = gen._sample_X(n)

    # Extract X columns from df in the same order
    xcols = names
    X_func = df[xcols].to_numpy()

    assert X_func.shape == X_class.shape
    # Values should be identical because both sampling use the same RNG seeded identically
    assert np.allclose(X_func, X_class)


def test_determinism_same_seed():
    seed = 777
    df1 = generate_rct(n=5000, split=0.55, random_state=seed, k=3, add_ancillary=False, target_type="binary")
    df2 = generate_rct(n=5000, split=0.55, random_state=seed, k=3, add_ancillary=False, target_type="binary")

    # Compare t, y, and X columns
    assert np.array_equal(df1["t"].to_numpy(), df2["t"].to_numpy())
    assert np.array_equal(df1["y"].to_numpy(), df2["y"].to_numpy())
    xcols = [c for c in df1.columns if c.startswith("x")]
    for c in xcols:
        assert np.array_equal(df1[c].to_numpy(), df2[c].to_numpy())


def test_ancillary_columns_present():
    df = generate_rct(n=1000, split=0.5, random_state=1, add_ancillary=True)
    ancillary = {"user_id", "age", "cnt_trans", "platform_Android", "platform_iOS", "invited_friend"}
    assert ancillary.issubset(df.columns)
