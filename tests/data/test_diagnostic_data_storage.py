import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalkit.data.causaldata import CausalData
from causalkit.inference.ate.dml_ate import dml_ate
from causalkit.inference.atte.dml_atte import dml_atte


def _make_synth(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    D = rng.binomial(1, p)
    Y = 2.0 * D + X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({
        "y": Y,
        "d": D,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
    })
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2", "x3"])
    return data, n


def test_dml_ate_returns_diagnostic_data():
    data, n = _make_synth(n=150, seed=123)
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000, solver="lbfgs")

    res = dml_ate(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        score="ATE",
        confidence_level=0.9,
        normalize_ipw=False,
        trimming_threshold=1e-2,
        random_state=7,
    )

    assert "diagnostic_data" in res, "dml_ate must return diagnostic_data key"
    dd = res["diagnostic_data"]
    for key in ["m_hat", "g0_hat", "g1_hat", "y", "d", "score", "normalize_ipw", "trimming_threshold", "p1"]:
        assert key in dd, f"diagnostic_data missing '{key}'"

    assert len(dd["m_hat"]) == n
    assert len(dd["g0_hat"]) == n
    assert len(dd["g1_hat"]) == n
    assert len(dd["y"]) == n
    assert len(dd["d"]) == n

    # Quick sanity check: propensity near-edges share is a valid probability
    eps = 0.01
    share_below = float(np.mean(dd["m_hat"] < eps))
    share_above = float(np.mean(dd["m_hat"] > 1 - eps))
    assert 0.0 <= share_below <= 1.0
    assert 0.0 <= share_above <= 1.0


def test_dml_att_returns_diagnostic_data():
    data, n = _make_synth(n=160, seed=321)
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000, solver="lbfgs")

    res = dml_atte(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        confidence_level=0.9,
        normalize_ipw=False,
        trimming_threshold=1e-2,
        random_state=11,
    )

    assert "diagnostic_data" in res, "dml_att must return diagnostic_data key"
    dd = res["diagnostic_data"]
    for key in ["m_hat", "g0_hat", "g1_hat", "y", "d", "score", "normalize_ipw", "trimming_threshold", "p1"]:
        assert key in dd, f"diagnostic_data missing '{key}'"

    assert dd["score"] == "ATTE"
    assert len(dd["m_hat"]) == n
    assert len(dd["y"]) == n
    assert len(dd["d"]) == n

    # ATT identity (raw): sum_controls m/(1-m) approx sum_treated 1
    d = dd["d"].astype(int)
    m = dd["m_hat"]
    lhs = np.sum((1 - d) * (m / (1 - m)))
    rhs = np.sum(d)
    # Not an equality in finite sample, but should be within a reasonable range
    rel_err = abs(lhs - rhs) / max(rhs, 1.0)
    assert rel_err < 0.5, "ATT raw identity too far off on synthetic data"