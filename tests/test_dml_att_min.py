import pytest
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data import CausalData
from causalkit.inference.att import dml_att_source


def _make_synthetic(n=300, seed=123):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    # Treatment assignment (logit based on x1, x2)
    logits = 0.5 * x1 + 0.8 * x2 - 0.2
    p = 1 / (1 + np.exp(-logits))
    t = rng.binomial(1, p, size=n)
    # Outcome (continuous): baseline + tau * t + noise
    tau = 1.0
    y = 0.3 * x1 + 0.7 * x2 + tau * t + rng.normal(0, 1.0, size=n)

    df = pd.DataFrame({
        "y": y.astype(float),
        "t": t.astype(int),
        "x1": x1.astype(float),
        "x2": x2.astype(float),
    })
    return df


def test_dml_att_runs_with_confounders_and_sklearn_models():
    df = _make_synthetic(n=300)
    data = CausalData(df=df, treatment="t", outcome="y", confounders=["x1", "x2"])

    # Lightweight sklearn models to avoid CatBoost dependency
    ml_g = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0)

    res = dml_att_source(data, ml_g=ml_g, ml_m=ml_m, n_folds=2, n_rep=1, confidence_level=0.9)

    # Basic sanity checks
    assert isinstance(res, dict)
    for key in ["coefficient", "std_error", "p_value", "confidence_interval", "model"]:
        assert key in res
    assert isinstance(res["coefficient"], float)
    assert isinstance(res["std_error"], float)
    assert isinstance(res["p_value"], float)
    assert isinstance(res["confidence_interval"], tuple)
