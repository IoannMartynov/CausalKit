import numpy as np

from causalkit.data.generators import CausalDatasetGenerator, generate_rct


def test_generate_emits_new_and_legacy_names_equal_class():
    gen = CausalDatasetGenerator(k=3, outcome_type="continuous", seed=0)
    df = gen.generate(1000)
    # New names present
    assert {"m", "g0", "g1", "cate"}.issubset(df.columns)
    # Legacy aliases present and equal
    assert np.allclose(df["m"].to_numpy(), df["propensity"].to_numpy())
    assert np.allclose(df["g0"].to_numpy(), df["mu0"].to_numpy())
    assert np.allclose(df["g1"].to_numpy(), df["mu1"].to_numpy())


def test_generate_rct_emits_aliases_equal():
    df = generate_rct(n=2000, split=0.4, random_state=123, target_type="binary", add_ancillary=False)
    assert {"m", "g0", "g1", "cate", "propensity", "mu0", "mu1"}.issubset(df.columns)
    assert np.allclose(df["m"].to_numpy(), df["propensity"].to_numpy())
    assert np.allclose(df["g0"].to_numpy(), df["mu0"].to_numpy())
    assert np.allclose(df["g1"].to_numpy(), df["mu1"].to_numpy())
