import numpy as np
import pandas as pd

from causalkit.data.causaldata import CausalData
from causalkit.refutation.score.score_validation import refute_irm_orthogonality


class DummyModel:
    def __init__(self, predictions, score='ATTE'):
        self.predictions = predictions
        self.score = score


def _dummy_inference_att(data: CausalData, **kwargs):
    df = data.get_df()
    confs = list(data.confounders)
    X = df[confs].to_numpy(dtype=float) if confs else np.zeros((len(df), 0))
    # Simple stable propensity and outcomes
    lin = X[:, 0] if X.shape[1] > 0 else np.zeros(len(df))
    g = 1 / (1 + np.exp(-lin))
    g = np.clip(g, 0.1, 0.9)
    m0 = 0.2 * lin
    m1 = m0 + 0.5

    y = df[data.target.name].astype(float).to_numpy()
    d = df[data.treatment.name].astype(float).to_numpy()

    preds = {
        'ml_m': g,
        'ml_g0': m0,
        'ml_g1': m1,
    }
    theta = float(y[d == 1].mean() - m0[d == 1].mean()) if (d == 1).any() else 0.0
    se = float(np.std(y - (d * m1 + (1 - d) * m0), ddof=1) / np.sqrt(len(df))) if len(df) > 1 else 0.0
    return {
        'coefficient': theta,
        'std_error': se,
        'model': DummyModel(predictions=preds, score='ATTE')
    }


def make_nullable_float_data(n=80, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Create nullable Float64 pandas dtype columns
    x1_s = pd.Series(x1, dtype="Float64")
    x2_s = pd.Series(x2, dtype="Float64")
    g = 1 / (1 + np.exp(-(x1 - 0.5 * x2)))
    d = rng.binomial(1, p=np.clip(g, 0.1, 0.9)).astype(float)
    m0 = 0.3 * x1 - 0.2 * x2
    y = m0 + 0.8 * d + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({
        'y': pd.Series(y, dtype='Float64'),
        'd': pd.Series(d, dtype='Float64'),
        'x1': x1_s,
        'x2': x2_s,
    })
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])


def test_refute_irm_orthogonality_with_nullable_float_confounders_runs():
    data = make_nullable_float_data(n=80, seed=13)
    res = refute_irm_orthogonality(
        _dummy_inference_att,
        data,
        target='ATTE',
        n_folds_oos=3,
        n_basis_funcs=3,
    )
    # Basic sanity of outputs
    assert 'oos_moment_test' in res
    assert 'orthogonality_derivatives' in res
    assert 'influence_diagnostics' in res
    assert 'theta' in res
