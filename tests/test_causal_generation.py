# tests/test_causal_generation.py
# Comprehensive tests for generators and related utilities

import numpy as np
import pandas as pd
import pytest

from causalkit.data.generators import (
    CausalDatasetGenerator,
    generate_rct,
    _sigmoid,
    _logit,
    _gaussian_copula,
    _sample_confounders_like_class,
)


# -------------------------
# Low-level utilities
# -------------------------

def test_sigmoid_stability_and_bounds():
    z = np.array([-1e6, -100.0, -10.0, 0.0, 10.0, 100.0, 1e6], dtype=float)
    s = _sigmoid(z)
    assert np.all(np.isfinite(s))
    assert np.all((s > 0.0) & (s < 1.0))
    assert np.all(np.diff(s) > 0)  # monotone increasing
    assert s[0] < 1e-12 and 1 - s[-1] < 1e-12


@pytest.mark.parametrize("p", [1e-9, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1 - 1e-9])
def test_logit_sigmoid_inverse(p):
    rec = _sigmoid(_logit(p))
    assert abs(rec - p) < 1e-9


# -------------------------
# RCT wrapper basic behavior
# -------------------------

@pytest.mark.parametrize(
    "target_type,target_params,split,mean_tol",
    [
        ("binary", {"p": {"A": 0.10, "B": 0.12}}, 0.5, 0.015),
        ("normal", {"mean": {"A": 0.00, "B": 0.20}, "std": 1.0}, 0.6, 0.03),
        ("poisson", {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}, 0.4, 0.05),
    ],
)
def test_generate_rct_group_means_and_split(target_type, target_params, split, mean_tol):
    n = 20000
    df = generate_rct(
        n=n,
        split=split,
        random_state=123,
        target_type=target_type,
        target_params=target_params,
        add_ancillary=True,
        deterministic_ids=True,
    )
    assert {"y", "t", "propensity", "mu0", "mu1", "cate"}.issubset(df.columns)

    # Check treatment split
    t_rate = df["t"].mean()
    assert abs(t_rate - split) < 0.02

    # Check group means on natural scale
    m0 = df.loc[df.t == 0, "y"].mean()
    m1 = df.loc[df.t == 1, "y"].mean()
    delta = m1 - m0

    if target_type == "binary":
        assert abs(m0 - 0.10) < mean_tol
        assert abs(m1 - 0.12) < mean_tol
    elif target_type == "normal":
        assert abs(delta - 0.20) < mean_tol
    else:
        # lamA=2.0, lamB=2.2 -> diff ~= 0.2
        assert abs(delta - 0.20) < mean_tol


def test_generate_rct_ancillaries_and_ids_determinism():
    n = 5000
    df1 = generate_rct(
        n=n,
        split=0.5,
        random_state=777,
        target_type="binary",
        add_ancillary=True,
        deterministic_ids=True,
    )
    df2 = generate_rct(
        n=n,
        split=0.5,
        random_state=777,
        target_type="binary",
        add_ancillary=True,
        deterministic_ids=True,
    )
    # All columns identical with deterministic IDs
    pd.testing.assert_frame_equal(df1, df2)

    # Without deterministic IDs: only 'user_id' should differ
    df3 = generate_rct(
        n=n,
        split=0.5,
        random_state=777,
        target_type="binary",
        add_ancillary=True,
        deterministic_ids=False,
    )
    assert not df1["user_id"].equals(df3["user_id"])
    pd.testing.assert_frame_equal(
        df1.drop(columns=["user_id"]), df3.drop(columns=["user_id"])
    )

    # Sanity on ancillary dtypes
    assert df1["age"].dtype.kind in "iu"
    assert df1["cnt_trans"].dtype.kind in "iu"
    assert set(np.unique(df1["platform_Android"])) <= {0, 1}
    assert set(np.unique(df1["platform_iOS"])) <= {0, 1}
    assert set(np.unique(df1["invited_friend"])) <= {0, 1}


# -------------------------
# Class generator behaviors
# -------------------------

def test_calibration_target_t_rate_simple():
    gen = CausalDatasetGenerator(
        k=0, outcome_type="continuous", theta=0.0, sigma_y=1.0, target_t_rate=0.3, seed=42
    )
    df = gen.generate(20000)
    assert abs(df["propensity"].mean() - 0.3) < 1e-3
    assert abs(df["t"].mean() - 0.3) < 0.02


def test_propensity_sharpness_changes_variation():
    # Same seed & structure, different sharpness -> different propensity dispersion
    common = dict(
        outcome_type="binary", alpha_y=0.0, theta=0.3, k=5, seed=123, target_t_rate=0.5
    )
    bt = np.array([0.6, -0.4, 0.3, 0.2, -0.5])
    gen_soft = CausalDatasetGenerator(beta_t=bt, propensity_sharpness=0.5, **common)
    gen_sharp = CausalDatasetGenerator(beta_t=bt, propensity_sharpness=3.0, **common)

    df_soft = gen_soft.generate(20000)
    df_sharp = gen_sharp.generate(20000)

    # Means will be ~0.5 due to calibration; dispersion should differ
    assert df_sharp["propensity"].std() > df_soft["propensity"].std()


def test_binary_outputs_well_formed():
    gen = CausalDatasetGenerator(
        k=3, outcome_type="binary", theta=0.0, seed=321, target_t_rate=0.5
    )
    df = gen.generate(5000)
    assert set(np.unique(df["y"])) <= {0.0, 1.0}
    assert np.all((df["mu0"].values > 0) & (df["mu0"].values < 1))
    assert np.all((df["mu1"].values > 0) & (df["mu1"].values < 1))
    assert np.all((df["propensity"].values > 0) & (df["propensity"].values < 1))


def test_poisson_overflow_guard():
    # Push link high; clip should prevent inf/nan
    gen = CausalDatasetGenerator(
        k=2,
        outcome_type="poisson",
        alpha_y=15.0,
        theta=5.0,
        seed=7,
        beta_y=np.array([2.0, 2.0]),
        beta_t=np.array([0.0, 0.0]),
        target_t_rate=0.5,
    )
    df = gen.generate(2000)
    for col in ["y", "mu0", "mu1"]:
        assert np.all(np.isfinite(df[col].values))


# -------------------------
# Confounders & copula
# -------------------------

def test_sample_confounders_like_class_categorical_onlylevel():
    rng = np.random.default_rng(1)
    specs = [
        {"name": "z", "dist": "categorical", "categories": ["A"], "probs": [1.0]}
    ]
    X, names = _sample_confounders_like_class(
        n=100, rng=rng, confounder_specs=specs, k=0, x_sampler=None, seed=1
    )
    assert names == ["z__onlylevel"]
    assert X.shape == (100, 1)
    assert np.all(X == 0.0)


def test_gaussian_copula_normal_corr_and_categorical_freqs():
    rng = np.random.default_rng(0)
    specs = [
        {"name": "a", "dist": "normal", "mu": 0.0, "sd": 1.0},
        {"name": "b", "dist": "normal", "mu": 0.0, "sd": 1.0},
        {
            "name": "c",
            "dist": "categorical",
            "categories": [0, 1, 2],
            "probs": [0.2, 0.3, 0.5],
        },
    ]
    corr = np.array([[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]])
    X, names = _gaussian_copula(rng, n=5000, specs=specs, corr=corr)
    # Names: a, b, c_1, c_2  (one-hot minus first level)
    assert names[:2] == ["a", "b"]
    assert set(names[2:]) == {"c_1", "c_2"}
    df = pd.DataFrame(X, columns=names)

    # Correlation between a and b close to 0.7 (tolerate approximation)
    rho = df[["a", "b"]].corr().iloc[0, 1]
    assert 0.6 < rho < 0.8

    # Categorical frequencies close to specified probs
    p1 = df["c_1"].mean()
    p2 = df["c_2"].mean()
    # Level 0 implicit prob = 1 - (p1+p2) ~ 0.2
    assert abs((1 - (p1 + p2)) - 0.2) < 0.03
    assert abs(p1 - 0.3) < 0.03
    assert abs(p2 - 0.5) < 0.03


# -------------------------
# Oracle nuisances
# -------------------------

def test_oracle_e_matches_monte_carlo():
    # Non-trivial treatment score w/ unobserved confounder
    bt = np.array([0.8, -0.3, 0.5])
    gen = CausalDatasetGenerator(
        beta_t=bt,
        k=3,
        seed=202,
        target_t_rate=0.4,
        outcome_type="continuous",
        theta=0.0,
        u_strength_t=0.7,
    )
    # Fix X and pick one row
    df = gen.generate(2000)
    x_cols = [
        c
        for c in df.columns
        if c not in {"y", "t", "m", "g0", "g1", "cate", "propensity", "mu0", "mu1"}
    ]
    x_row = df[x_cols].iloc[0].to_numpy(dtype=float)

    e_marginal, _, _ = gen.oracle_nuisance(num_quad=25)
    e_quad = e_marginal(x_row)

    # MC approx over U ~ N(0,1)
    rng = np.random.default_rng(99)
    U = rng.normal(size=20000)
    score = gen.alpha_t + gen._treatment_score(
        x_row.reshape(1, -1), np.zeros(1, dtype=float)
    )[0]
    e_mc = _sigmoid(score + gen.u_strength_t * U).mean()

    assert abs(e_quad - e_mc) < 0.02


# -------------------------
# to_causal_data
# -------------------------

def test_to_causal_data_defaults_and_types():
    CausalData = pytest.importorskip("causalkit.data.causaldata").CausalData  # type: ignore
    gen = CausalDatasetGenerator(k=4, outcome_type="continuous", seed=11)
    cd = gen.to_causal_data(n=1000)
    assert isinstance(cd, CausalData)

    # Confounders should exclude ground-truth/meta cols and preserve order
    df = cd.df  # assuming CausalData exposes .df; if not, adapt this check
    exclude = {"y", "t", "m", "g0", "g1", "cate", "propensity", "mu0", "mu1"}
    expected = [c for c in df.columns if c not in exclude]
    # If CausalData has attribute 'confounders', verify ordering
    if hasattr(cd, "confounders"):
        assert list(cd.confounders) == expected
