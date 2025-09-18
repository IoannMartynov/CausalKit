import numpy as np

from causalkit.data.generators import CausalDatasetGenerator, _sigmoid


def test_oracle_gating_raises_when_U_affects_both():
    # When U impacts both treatment and outcome, DML is not identified
    gen = CausalDatasetGenerator(
        k=2,
        beta_t=np.array([0.5, -0.2], dtype=float),
        beta_y=np.array([0.3, 0.1], dtype=float),
        alpha_t=0.1,
        alpha_y=0.0,
        u_strength_t=0.7,
        u_strength_y=0.4,
        outcome_type="continuous",
        seed=123,
    )
    with np.testing.assert_raises(ValueError):
        gen.oracle_nuisance()


def test_m0_m1_mappings_across_outcomes():
    x = np.array([0.2, -0.3], dtype=float)
    beta = np.array([1.0, -2.0], dtype=float)

    # Continuous
    gen_c = CausalDatasetGenerator(k=2, beta_y=beta, alpha_y=0.5, theta=0.7, outcome_type="continuous", seed=0)
    m_fun, g0_c, g1_c = gen_c.oracle_nuisance()
    loc0 = 0.5 + x @ beta + 0.0  # U=0
    loc1 = loc0 + 0.7
    assert abs(g0_c(x) - float(loc0)) < 1e-12
    assert abs(g1_c(x) - float(loc1)) < 1e-12

    # Binary
    gen_b = CausalDatasetGenerator(k=2, beta_y=beta, alpha_y=-0.4, theta=0.3, outcome_type="binary", seed=0)
    _, g0_b, g1_b = gen_b.oracle_nuisance()
    loc0 = -0.4 + x @ beta
    loc1 = loc0 + 0.3
    assert abs(g0_b(x) - float(_sigmoid(loc0))) < 1e-12
    assert abs(g1_b(x) - float(_sigmoid(loc1))) < 1e-12

    # Poisson (keep locations small to avoid clipping differences)
    gen_p = CausalDatasetGenerator(k=2, beta_y=beta, alpha_y=0.1, theta=0.2, outcome_type="poisson", seed=0)
    _, g0_p, g1_p = gen_p.oracle_nuisance()
    loc0 = 0.1 + x @ beta
    loc1 = loc0 + 0.2
    assert abs(g0_p(x) - float(np.exp(loc0))) < 1e-12
    assert abs(g1_p(x) - float(np.exp(loc1))) < 1e-12


def test_e_reduces_to_sigmoid_when_no_U():
    gen = CausalDatasetGenerator(k=2, beta_t=np.array([0.5, 0.1], dtype=float), alpha_t=-0.3, u_strength_t=0.0, seed=1)
    x = np.array([1.0, -2.0], dtype=float)

    m_fun, _, _ = gen.oracle_nuisance()
    base = gen._treatment_score(x.reshape(1, -1), np.zeros(1, dtype=float))[0]
    expected = _sigmoid(gen.alpha_t + base)
    assert abs(m_fun(x) - float(expected)) < 1e-12
