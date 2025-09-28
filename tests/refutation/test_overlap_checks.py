import numpy as np

from causalkit.refutation.overlap import (
    positivity_overlap_checks,
    edge_mass,
    ks_distance,
    auc_for_m,
    ess_per_group,
    att_weight_sum_identity,
)


def test_split_functions_basic():
    rng = np.random.default_rng(0)
    n = 500
    m = rng.beta(2, 2, size=n)
    D = rng.binomial(1, m)

    # Edge mass
    em = edge_mass(m, eps=(0.01, 0.02))
    assert 0.0 <= em[0.01]["share_below"] <= 1.0
    assert 0.0 <= em[0.02]["share_above"] <= 1.0

    # KS and AUC
    ks = ks_distance(m, D)
    auc = auc_for_m(m, D)
    assert (np.isnan(ks) or 0.0 <= ks <= 1.0)
    assert (np.isnan(auc) or 0.0 <= auc <= 1.0)

    # ESS
    ess = ess_per_group(m, D)
    assert set(["ess_w1", "ess_w0", "ess_ratio_w1", "ess_ratio_w0"]).issubset(ess.keys())

    # ATT identity
    att_id = att_weight_sum_identity(m, D)
    assert set(["lhs_sum", "rhs_sum", "rel_err"]).issubset(att_id.keys())


def test_positivity_overlap_checks_returns_dict_and_flags():
    # Make nearly separable data to trigger strong flags
    rng = np.random.default_rng(1)
    n = 400
    x = rng.standard_normal(n)
    D = (x > 0).astype(int)
    m = 1 / (1 + np.exp(-5 * x))
    rep = positivity_overlap_checks(m_hat=m, D=D, m_clipped_from=(1e-3, 1 - 1e-3))
    assert isinstance(rep, dict)
    assert "flags" in rep and isinstance(rep["flags"], dict)
    red_any = any(v == "RED" for k, v in rep["flags"].items() if k in {"auc", "ks", "edge_mass_001", "edge_mass_002"})
    assert red_any
