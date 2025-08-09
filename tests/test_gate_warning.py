import types
import warnings
import numpy as np
import pandas as pd
import pytest

from causalkit.data.causaldata import CausalData
from causalkit.inference.gate import gate_esimand


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
        # Provide orthogonal signals after fit
        self._orthogonal_signals = None

    def fit(self):
        # Emit the sklearn-style FutureWarning we want to suppress
        warnings.warn(
            "'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
            category=FutureWarning,
            stacklevel=2,
        )
        # Create deterministic signals
        n = len(self.data_dml.df)
        rng = np.random.default_rng(0)
        # simple centered signals
        self._orthogonal_signals = rng.normal(size=n)

    # Do not define a 'gate' method to force fallback path in gate_esimand


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


def _make_data(n=80):
    rng = np.random.default_rng(1)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    y = 0.8 * t + 0.4 * X1 - 0.1 * X2 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"y": y, "t": t, "x1": X1, "x2": X2})
    return CausalData(df=df, treatment="t", outcome="y", cofounders=["x1", "x2"])


def test_gate_esimand_suppresses_force_all_finite_warning(monkeypatch, recwarn):
    # Import module and monkeypatch its 'dml' reference
    import importlib
    gate_module = importlib.import_module('causalkit.inference.gate.gate_esimand')

    fake_doubleml = types.SimpleNamespace(DoubleMLData=FakeDoubleMLData, DoubleMLIRM=FakeDoubleMLIRM)
    monkeypatch.setattr(gate_module, "dml", fake_doubleml, raising=True)

    data = _make_data()

    # Call gate_esimand and ensure FutureWarning about 'force_all_finite' is not propagated
    res = gate_esimand(data, n_groups=4, n_folds=2, n_rep=1, ml_g=StubRegressor(), ml_m=StubClassifier())

    assert isinstance(res, pd.DataFrame)
    assert set(["group", "n", "theta", "std_error", "p_value", "ci_lower", "ci_upper"]).issubset(res.columns)

    msgs = [str(w.message) for w in recwarn]
    assert not any("force_all_finite" in m for m in msgs), msgs
