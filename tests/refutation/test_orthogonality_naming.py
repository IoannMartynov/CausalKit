import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalkit.data.causaldata import CausalData
from causalkit.inference.estimators.irm import IRM
from causalkit.refutation.score.score_validation import (
    extract_nuisances,
    aipw_score_atte,
    aipw_score_att,
    refute_irm_orthogonality,
)


def _toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    m = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))
    d = rng.binomial(1, m)
    g0 = X[:, 0] + rng.normal(scale=0.1, size=n)
    g1 = X[:, 0] + 1.0 + rng.normal(scale=0.1, size=n)
    y = g0 * (1 - d) + g1 * d
    df = pd.DataFrame({"y": y, "d": d, "x1": X[:, 0], "x2": X[:, 1]})
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])
    return data


def _fit_irm(data, score="ATE"):
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000)
    irm = IRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=2, score=score)
    irm.fit()
    return irm


def _inference_fn(data, score="ATE"):
    model = _fit_irm(data, score=score)
    return {"model": model, "coefficient": float(model.coef_[0]), "std_error": float(model.se_[0])}


def test_extract_nuisances_order():
    data = _toy_data(120)
    irm = _fit_irm(data, score="ATE")
    m, g0, g1 = extract_nuisances(irm)
    assert m.shape == g0.shape == g1.shape
    # Propensity should be in (0,1)
    assert np.all((m > 0) & (m < 1))


def test_aipw_atte_wrapper_equivalence():
    n = 50
    rng = np.random.default_rng(1)
    y = rng.normal(size=n)
    d = rng.integers(0, 2, size=n)
    g0 = rng.normal(size=n)
    g1 = rng.normal(size=n)
    m = rng.uniform(0.05, 0.95, size=n)
    theta = 0.5
    psi_new = aipw_score_atte(y, d, g0, g1, m, theta, p_treated=float(d.mean()), trimming_threshold=0.01)
    psi_old = aipw_score_att(y, d, m0=g0, m1=g1, g=m, theta=theta, p1=float(d.mean()), eps=0.01)
    assert np.allclose(psi_new, psi_old)


def test_refute_irm_orthogonality_api_keys():
    data = _toy_data(150)
    res = refute_irm_orthogonality(lambda d: _inference_fn(d, score="ATTE"), data, score="ATTE")
    # New param keys
    assert res["params"]["score"] in ("ATE", "ATTE")
    assert "trimming_threshold" in res["params"]
    # New overlap key present for ATTE
    assert "overlap_atte" in res
    # Influence uses new naming internally; basic presence check
    assert "influence_diagnostics" in res
