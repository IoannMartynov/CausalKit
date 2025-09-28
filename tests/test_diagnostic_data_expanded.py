import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.ate import dml_ate
from causalkit.inference.att import dml_att


def make_causal_data(n=400, target_type="normal", random_state=12):
    df = generate_rct(n=n, split=0.5, random_state=random_state, target_type=target_type, k=3, add_ancillary=False)
    y = "y"; t = "t"
    xcols = [c for c in df.columns if c not in {y, t, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, t] + xcols], treatment=t, outcome=y, confounders=xcols)


def _assert_diag(d, n, k):
    # presence
    for key in [
        "m_hat", "g0_hat", "g1_hat", "y", "d", "x", "psi", "psi_a", "psi_b", "folds",
        "score", "normalize_ipw", "trimming_threshold", "p1"
    ]:
        assert key in d, f"Missing key {key} in diagnostic_data"

    # shapes
    assert d["m_hat"].shape[0] == n
    assert d["g0_hat"].shape[0] == n
    assert d["g1_hat"].shape[0] == n
    assert d["y"].shape[0] == n
    assert d["d"].shape[0] == n
    assert d["x"].shape == (n, k)
    assert d["psi"].shape[0] == n
    assert d["psi_a"].shape[0] == n
    assert d["psi_b"].shape[0] == n
    assert d["folds"].shape[0] == n

    # folds values are valid indices [0, n_folds-1]
    assert np.min(d["folds"]) >= 0
    assert np.all(np.isfinite(d["folds"]))

    # types
    assert isinstance(d["score"], str)
    assert isinstance(d["normalize_ipw"], (bool, np.bool_))
    float(d["trimming_threshold"])  # castable to float
    float(d["p1"])  # castable to float


def test_diagnostic_data_expanded_dml_ate():
    cd = make_causal_data(n=350, target_type="normal", random_state=7)
    k = len(cd.confounders)
    ml_g = RandomForestRegressor(n_estimators=40, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=40, random_state=0)

    res = dml_ate(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, score="ATE", normalize_ipw=True, trimming_threshold=1e-3)
    dd = res["diagnostic_data"]
    _assert_diag(dd, n=cd.df.shape[0], k=k)
    assert dd["score"] == "ATE"


def test_diagnostic_data_expanded_dml_att():
    cd = make_causal_data(n=360, target_type="binary", random_state=21)
    k = len(cd.confounders)
    ml_g = RandomForestClassifier(n_estimators=50, random_state=21)
    ml_m = RandomForestClassifier(n_estimators=50, random_state=21)

    res = dml_att(cd, ml_g=ml_g, ml_m=ml_m, n_folds=4, normalize_ipw=False, trimming_threshold=5e-3)
    dd = res["diagnostic_data"]
    _assert_diag(dd, n=cd.df.shape[0], k=k)
    assert dd["score"] == "ATTE"
