import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from causalis.refutation.overlap import att_overlap_tests
from causalis.data.causaldata import CausalData
from causalis.inference.atte.dml_atte import dml_atte


def test_att_overlap_tests_structure_and_basic_flags_random():
    rng = np.random.default_rng(123)
    n = 500
    # Simulate m in (0.05, 0.95) and D ~ Bernoulli(m)
    m = rng.uniform(0.05, 0.95, size=n)
    D = rng.binomial(1, m)

    res = {"diagnostic_data": {"m_hat": m, "d": D}}
    out = att_overlap_tests(res)

    # Structure
    assert isinstance(out, dict)
    assert "edge_mass" in out and "ks" in out and "auc" in out and "ess" in out and "att_weight_identity" in out
    assert isinstance(out["edge_mass"].get("eps"), dict)
    # KS should be defined (not NaN) as both arms present
    assert out["ks"]["value"] == out["ks"]["value"]  # not NaN
    assert isinstance(out["ks"]["warn"], bool)
    # AUC within [0,1]
    assert 0.0 <= out["auc"]["value"] <= 1.0
    assert out["auc"]["flag"] in {"GREEN", "YELLOW", "RED", "NA"}

    # ESS entries
    tr = out["ess"]["treated"]
    ct = out["ess"]["control"]
    for side in (tr, ct):
        assert set(["ess", "n", "ratio", "flag"]).issubset(side.keys())
        assert side["n"] > 0
        assert 0.0 <= side["ratio"] <= 1.0
        assert side["flag"] in {"GREEN", "YELLOW", "RED", "NA"}

    # ATT identity: lhs approx rhs
    lhs = out["att_weight_identity"]["lhs_sum"]
    rhs = out["att_weight_identity"]["rhs_sum"]
    rel_err = out["att_weight_identity"]["rel_err"]
    assert abs(lhs - rhs) / max(rhs, 1e-12) - rel_err < 1e-12
    # Most random cases should be within 10%
    assert rel_err < 0.10


def test_att_overlap_tests_accepts_dml_att_result():
    # Small synthetic dataset with overlap
    rng = np.random.default_rng(202)
    n = 300
    X0 = rng.normal(size=n)
    X1 = rng.normal(size=n)
    logits = 0.8 * X0 - 0.4 * X1
    m_true = 1 / (1 + np.exp(-logits))
    D = rng.binomial(1, m_true)
    Y = 1.5 * D + X0 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"x0": X0, "x1": X1, "D": D, "Y": Y})

    data = CausalData(df=df, treatment="D", outcome="Y", confounders=["x0", "x1"])
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=2000)

    res = dml_atte(data, ml_g=ml_g, ml_m=ml_m, n_folds=4, trimming_threshold=1e-3, random_state=7)

    out = att_overlap_tests(res)
    # Sanity checks
    assert isinstance(out, dict)
    assert out["auc"]["flag"] in {"GREEN", "YELLOW", "RED", "NA"}
    assert isinstance(out["ks"]["warn"], bool)
    # Identity should not be wildly off
    assert out["att_weight_identity"]["rel_err"] < 0.20
