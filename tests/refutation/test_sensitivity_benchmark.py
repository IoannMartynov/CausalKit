import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalis.data.causaldata import CausalData
from causalis.inference.estimators.irm import IRM
from causalis.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_benchmark


def make_synthetic(n=400, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Treatment depends on x1 strongly, x2 weakly
    logits = 1.0 * x1 + 0.2 * x2
    p = 1 / (1 + np.exp(-logits))
    d = rng.binomial(1, p)
    # Outcome depends on d and x1
    y = 1.0 * d + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return df


def fit_irm(df):
    data = CausalData(df, treatment="d", outcome="y", confounders=["x1", "x2"])
    ml_g = RandomForestRegressor(n_estimators=50, random_state=1)
    ml_m = LogisticRegression(max_iter=1000)
    irm = IRM(data=data, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=1)
    irm.fit()
    return irm


def test_sensitivity_benchmark_basic():
    df = make_synthetic()
    irm = fit_irm(df)
    effect = {"model": irm}

    # Remove the strong confounder x1
    res = sensitivity_benchmark(effect, ["x1"])  # returns DataFrame

    assert hasattr(res, "loc")  # is DataFrame-like
    assert res.shape[0] == 1
    # expected columns
    for col in ["cf_y", "cf_d", "rho", "theta_long", "theta_short", "delta"]:
        assert col in res.columns

    # Delta should be non-zero when removing strong confounder
    assert abs(float(res["delta"].iloc[0])) > 0.0


def test_input_validation():
    df = make_synthetic()
    irm = fit_irm(df)
    effect = {"model": irm}

    with pytest.raises(TypeError):
        sensitivity_benchmark(None, ["x1"])  # type: ignore
    with pytest.raises(TypeError):
        sensitivity_benchmark(effect, "x1")  # type: ignore
    with pytest.raises(ValueError):
        sensitivity_benchmark(effect, [])
    with pytest.raises(ValueError):
        sensitivity_benchmark(effect, ["not_in_X"])  # not a confounder
