import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalkit.data.causaldata import CausalData
from causalkit.inference.ate import dml_ate
from causalkit.inference.att import dml_att


def _make_synth(n=120, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    D = rng.binomial(1, p)
    Y = 1.5 * D + X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({
        "y": Y,
        "d": D,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
    })
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2", "x3"])
    return data


def test_dml_ate_store_diagnostic_data_flag():
    data = _make_synth(n=140, seed=99)
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000)

    # Default: should include diagnostic_data and not None
    res_default = dml_ate(data, ml_g=ml_g, ml_m=ml_m, n_folds=3)
    assert "diagnostic_data" in res_default
    assert res_default["diagnostic_data"] is not None

    # Flag off: should include key with None to allow downstream code to check
    res_off = dml_ate(data, ml_g=ml_g, ml_m=ml_m, n_folds=3, store_diagnostic_data=False)
    assert "diagnostic_data" in res_off
    assert res_off["diagnostic_data"] is None


def test_dml_att_store_diagnostic_data_flag():
    data = _make_synth(n=150, seed=77)
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000)

    # Default True
    res_default = dml_att(data, ml_g=ml_g, ml_m=ml_m, n_folds=3)
    assert "diagnostic_data" in res_default
    assert res_default["diagnostic_data"] is not None

    # False
    res_off = dml_att(data, ml_g=ml_g, ml_m=ml_m, n_folds=3, store_diagnostic_data=False)
    assert "diagnostic_data" in res_off
    assert res_off["diagnostic_data"] is None
