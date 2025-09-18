import numpy as np

import pandas as pd

from causalkit.data.generators import generate_rct, CausalDatasetGenerator


def test_generate_rct_wrapper_binary_sets_mu_and_propensity_constant():
    n = 5000
    split = 0.3
    params = {"p": {"A": 0.20, "B": 0.55}}
    df = generate_rct(
        n=n,
        split=split,
        random_state=123,
        target_type="binary",
        target_params=params,
        confounder_specs=[{"name": "z", "dist": "normal", "mu": 0.0, "sd": 1.0}],
        add_ancillary=False,
    )
    # Propensity should be constant and equal to split
    prop = df["propensity"].to_numpy()
    assert float(prop.max() - prop.min()) < 1e-12
    assert abs(float(prop[0]) - split) < 1e-4
    # mu0/mu1 constants on natural scale
    assert np.allclose(df["mu0"].unique(), [params["p"]["A"]], atol=1e-6)
    assert np.allclose(df["mu1"].unique(), [params["p"]["B"]], atol=1e-6)
    # y and t should be float type
    assert df["y"].dtype.kind == "f"
    assert df["t"].dtype.kind == "f"
    # y,t in {0.0,1.0}
    assert set(np.unique(df["t"])) <= {0.0, 1.0}
    assert set(np.unique(df["y"])) <= {0.0, 1.0}


def test_degenerate_categorical_unique_name_in_class_and_copula():
    specs = [{"name": "cat", "dist": "categorical", "categories": [1]}]
    # Plain class sampling
    gen = CausalDatasetGenerator(confounder_specs=specs, outcome_type="continuous", seed=0)
    d1 = gen.generate(10)
    assert "cat__onlylevel" in d1.columns
    assert (d1["cat__onlylevel"].to_numpy() == 0.0).all()
    # Copula path
    gen2 = CausalDatasetGenerator(confounder_specs=specs, use_copula=True, copula_corr=np.array([[1.0]]), outcome_type="continuous", seed=0)
    d2 = gen2.generate(10)
    assert "cat__onlylevel" in d2.columns
    assert (d2["cat__onlylevel"].to_numpy() == 0.0).all()


def test_propensity_sharpness_increases_extreme_propensities():
    # Create informative X so beta_t has an effect
    specs = [
        {"name": f"x{i}", "dist": "normal", "mu": 0.0, "sd": 1.0} for i in range(5)
    ]
    beta = np.ones(5, dtype=float)
    n = 20000
    gen_soft = CausalDatasetGenerator(
        confounder_specs=specs,
        beta_t=beta,
        outcome_type="continuous",
        seed=7,
        propensity_sharpness=1.0,
    )
    df_soft = gen_soft.generate(n)

    gen_sharp = CausalDatasetGenerator(
        confounder_specs=specs,
        beta_t=beta,
        outcome_type="continuous",
        seed=7,
        propensity_sharpness=5.0,
    )
    df_sharp = gen_sharp.generate(n)

    def frac_extreme(p):
        p = p.to_numpy()
        return float(((p < 0.1) | (p > 0.9)).mean())

    f_soft = frac_extreme(df_soft["propensity"]) 
    f_sharp = frac_extreme(df_sharp["propensity"]) 

    assert f_sharp > f_soft + 0.05  # noticeably more extreme with higher sharpness
