import types
import warnings
import numpy as np
import pandas as pd
import pytest

from causalkit.data.causaldata import CausalData


class FakeDoubleMLData:
    def __init__(self, df, y_col, d_cols, x_cols):
        self.df = df
        self.y_col = y_col
        self.d_cols = d_cols
        self.x_cols = x_cols


class FakeDoubleMLIRM:
    def __init__(self, data_dml, ml_g, ml_m, n_folds, n_rep, score):
        self.data_dml = data_dml
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.score = score
        self.coef = np.array([1.0])
        self.se = np.array([0.1])
        self.pval = np.array([0.05])

    def fit(self):
        warnings.warn(
            "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
            category=FutureWarning,
            stacklevel=2,
        )

    def confint(self, level=0.95):
        # return numpy array
        return np.array([[0.8, 1.2]])


class StubRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class StubClassifier:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


def _make_data(n=50):
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    y = 1.0 * t + 0.5 * X1 - 0.2 * X2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y": y, "t": t, "x1": X1, "x2": X2})
    return CausalData(df=df, treatment="t", outcome="y", cofounders=["x1", "x2"])


def test_dml_suppresses_force_all_finite_warning(monkeypatch, recwarn):
    # Import module and monkeypatch its doubleml reference
    import importlib
    dml_module = importlib.import_module('causalkit.inference.att.dml')
    from causalkit.inference.att import dml as dml_func

    fake_doubleml = types.SimpleNamespace(DoubleMLData=FakeDoubleMLData, DoubleMLIRM=FakeDoubleMLIRM)
    monkeypatch.setattr(dml_module, "doubleml", fake_doubleml, raising=True)

    data = _make_data()

    # Call dml and ensure no FutureWarning about 'force_all_finite' is propagated
    res = dml_func(data, ml_g=StubRegressor(), ml_m=StubClassifier(), n_folds=3, n_rep=1, confidence_level=0.95)

    assert isinstance(res, dict)
    assert "coefficient" in res and "confidence_interval" in res

    # Ensure warning with the deprecation message is not present in recorded warnings
    msgs = [str(w.message) for w in recwarn]
    assert not any("force_all_finite" in m for m in msgs), msgs
