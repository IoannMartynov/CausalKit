import numpy as np
import pandas as pd

from causalis.refutation.score.score_validation import (
    aipw_score_atte,
    orthogonality_derivatives_atte,
    refute_irm_orthogonality,
)
from causalis.data.causaldata import CausalData


class DummyModel:
    def __init__(self, predictions, score='ATTE'):
        self.predictions = predictions
        self.score = score


def dummy_inference_fn(data: CausalData, **kwargs):
    df = data.get_df()
    # Simple predictions based on confounders
    X = df[list(data.confounders)].values if data.confounders else np.zeros((len(df), 1))
    # Propensity bounded away from 0/1
    g = 1 / (1 + np.exp(-X[:, 0])) if X.shape[1] > 0 else np.full(len(df), 0.5)
    g = np.clip(g, 0.1, 0.9)
    # Outcome models m0, m1
    m0 = (X[:, 0] if X.shape[1] > 0 else 0) * 0.5
    m1 = m0 + 1.0
    preds = {
        'ml_m': g,
        'ml_g0': m0,
        'ml_g1': m1,
    }
    # Simple ATT estimate: E[Y|D=1] - E[m0(X)|D=1]
    y = df[data.target.name].values.astype(float)
    d = df[data.treatment.name].values.astype(float)
    theta = float(y[d == 1].mean() - m0[d == 1].mean()) if (d == 1).any() else 0.0
    se = float(np.std(y - (d * m1 + (1 - d) * m0), ddof=1) / np.sqrt(len(df))) if len(df) > 1 else 0.0
    return {
        'coefficient': theta,
        'std_error': se,
        'model': DummyModel(predictions=preds, score='ATTE')
    }


def make_data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    g = 1 / (1 + np.exp(-(x1 - 0.5 * x2)))
    d = rng.binomial(1, p=np.clip(g, 0.05, 0.95))
    m0 = 0.5 * x1 - 0.3 * x2
    tau = 1.0
    y = m0 + tau * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])


def test_aipw_score_atte_sign_and_norm():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([1.0, 0.0, 1.0, 0.0])
    m0 = np.array([0.5, 0.5, 0.5, 0.5])
    m1 = np.array([1.5, 1.5, 1.5, 1.5])
    g = np.array([0.2, 0.8, 0.3, 0.9])
    theta = 0.7
    p1 = d.mean()
    gamma = g / (1 - g)
    expected = (d * (y - m0 - theta) - (1 - d) * gamma * (y - m0)) / (p1 + 1e-12)
    got = aipw_score_atte(y, d, g0=m0, g1=m1, m=g, theta=theta, p_treated=p1, trimming_threshold=1e-6)
    assert np.allclose(got, expected)


def test_orthogonality_derivatives_atte_structure():
    n, B = 50, 3
    rng = np.random.default_rng(0)
    X_basis = rng.normal(size=(n, B))
    y = rng.normal(size=n)
    d = rng.binomial(1, 0.4, size=n).astype(float)
    m0 = rng.normal(size=n)
    g = np.clip(rng.uniform(0.1, 0.9, size=n), 0.1, 0.9)
    p1 = float(d.mean())
    df = orthogonality_derivatives_atte(X_basis, y, d, g0=m0, m=g, p_treated=p1, trimming_threshold=1e-6)
    assert len(df) == B
    for col in ['d_g1', 'se_g1', 't_g1', 'd_g0', 'se_g0', 't_g0', 'd_m', 'se_m', 't_m']:
        assert col in df.columns
    # Under ATTE, derivative wrt g1 is zero
    assert np.allclose(df['t_g1'].values, 0.0)


def test_refute_irm_orthogonality_att_outputs():
    data = make_data(n=120, seed=42)
    res = refute_irm_orthogonality(
        dummy_inference_fn, data,
        score='ATTE', n_folds_oos=3, n_basis_funcs=3, trim_propensity=(0.02, 0.98), strict_oos=True
    )
    # Ensure ATTE score selected
    assert res['params']['score'] in ('ATTE', 'ATT')
    # p_treated family should be in params for ATTE
    assert 'p_treated' in res['params'] and 0 < res['params']['p_treated'] < 1
    assert 'p_treated_full' in res['params'] and 0 < res['params']['p_treated_full'] < 1
    assert 'p_treated_trim' in res['params'] and 0 < res['params']['p_treated_trim'] < 1
    # Overlap diagnostics present with m-based column names
    assert 'overlap_atte' in res and res['overlap_atte'] is not None
    overlap = res['overlap_atte']
    assert all(col in overlap.columns for col in [
        'pct_controls_with_m_ge_thr', 'pct_treated_with_m_le_1_minus_thr'
    ])
    # Robustness curve present (may occasionally be None if inference fails; ensure key exists)
    assert 'robustness' in res and 'trim_curve_atte' in res['robustness']
    # Derivatives for ATTE should include IRM naming columns
    ortho = res['orthogonality_derivatives']['full_sample']
    assert isinstance(ortho, pd.DataFrame)
    if len(ortho) > 0:
        for col in ['t_g0', 't_m']:
            assert col in ortho.columns
    # Strict OOS stats present and selected
    oos = res['oos_moment_test']
    assert 'tstat_strict' in oos and 'pvalue_strict' in oos
    assert res['params']['strict_oos_applied'] is True
