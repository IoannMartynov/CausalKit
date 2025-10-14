import numpy as np

from causalis.data.generators import CausalDatasetGenerator


def test_copula_respects_corr_for_normals():
    # Two normal marginals with target correlation 0.7
    specs = [
        {"name": "a", "dist": "normal", "mu": 0.0, "sd": 1.0},
        {"name": "b", "dist": "normal", "mu": 0.0, "sd": 1.0},
    ]
    corr = np.array([[1.0, 0.7], [0.7, 1.0]], dtype=float)

    gen = CausalDatasetGenerator(
        confounder_specs=specs,
        use_copula=True,
        copula_corr=corr,
        outcome_type="continuous",
        seed=123,
    )
    df = gen.generate(20000)
    r = np.corrcoef(df["a"].to_numpy(), df["b"].to_numpy())[0, 1]
    assert abs(r - 0.7) < 0.05


def test_copula_mixed_types_shapes_and_names():
    specs = [
        {"name": "x_norm", "dist": "normal", "mu": 5.0, "sd": 2.0},
        {"name": "x_unif", "dist": "uniform", "a": -1.0, "b": 3.0},
        {"name": "x_bin", "dist": "bernoulli", "p": 0.3},
        {"name": "x_cat", "dist": "categorical", "categories": [0, 1, 2]},
    ]
    corr = np.array(
        [
            [1.0, 0.4, 0.2, 0.0],
            [0.4, 1.0, 0.1, 0.0],
            [0.2, 0.1, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    gen = CausalDatasetGenerator(
        confounder_specs=specs,
        use_copula=True,
        copula_corr=corr,
        outcome_type="continuous",
        seed=0,
    )
    df = gen.generate(5000)
    # Expect 1 + 1 + 1 + (3-1) = 5 columns from specs
    expected_cols = ["x_norm", "x_unif", "x_bin", "x_cat_1", "x_cat_2"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c}"
    # One-hot sanity: no row has both 1s for the same categorical
    oh_sum = df[["x_cat_1", "x_cat_2"]].sum(axis=1)
    assert (oh_sum <= 1.0 + 1e-8).all()
    # Types sanity
    assert df["x_bin"].dropna().isin([0.0, 1.0]).all()


def test_use_copula_flag_gates_behavior():
    specs = [
        {"name": "z1", "dist": "normal", "mu": 0.0, "sd": 1.0},
        {"name": "z2", "dist": "normal", "mu": 0.0, "sd": 1.0},
    ]
    corr = np.array([[1.0, 0.8], [0.8, 1.0]], dtype=float)

    gen_indep = CausalDatasetGenerator(
        confounder_specs=specs,
        use_copula=False,
        outcome_type="continuous",
        seed=1,
    )
    df_i = gen_indep.generate(20000)
    r_i = np.corrcoef(df_i["z1"].to_numpy(), df_i["z2"].to_numpy())[0, 1]

    gen_cop = CausalDatasetGenerator(
        confounder_specs=specs,
        use_copula=True,
        copula_corr=corr,
        outcome_type="continuous",
        seed=1,
    )
    df_c = gen_cop.generate(20000)
    r_c = np.corrcoef(df_c["z1"].to_numpy(), df_c["z2"].to_numpy())[0, 1]

    assert abs(r_i) < 0.2
    assert abs(r_c - 0.8) < 0.05
