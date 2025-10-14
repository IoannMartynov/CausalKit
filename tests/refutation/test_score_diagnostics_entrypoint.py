import numpy as np
import pandas as pd

from causalis.refutation.score.score_validation import run_score_diagnostics


def _make_synth(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    beta0 = np.array([1.0, -0.5, 0.3])
    tau = 1.2
    # outcome regressions
    g0 = X @ beta0
    g1 = g0 + tau
    # propensity
    logits = X @ np.array([0.5, -0.2, 0.1])
    m = 1 / (1 + np.exp(-logits))
    d = rng.binomial(1, m).astype(float)
    y = g0 + d * tau + rng.normal(scale=1.0, size=n)
    # AIPW plug-in theta using true nuisances for stability
    theta_terms = (g1 - g0) + d * (y - g1) / np.clip(m, 1e-3, 1 - 1e-3) - (1 - d) * (y - g0) / np.clip(1 - m, 1e-3, 1.0)
    theta = float(theta_terms.mean())
    return y, d, g0, g1, m, theta


def test_run_score_diagnostics_with_arrays_ate():
    y, d, g0, g1, m, theta = _make_synth(seed=42, n=200)
    rep = run_score_diagnostics(y=y, d=d, g0=g0, g1=g1, m=m, theta=theta, score='ATE', return_summary=True)
    assert isinstance(rep, dict)
    assert 'influence_diagnostics' in rep
    assert 'orthogonality_derivatives' in rep
    # summary dataframe
    assert 'summary' in rep
    assert isinstance(rep['summary'], pd.DataFrame)
    # meta
    assert rep['meta']['n'] == len(y)
    assert rep['params']['score'] == 'ATE'


def test_run_score_diagnostics_with_result_dict():
    y, d, g0, g1, m, theta = _make_synth(seed=7, n=150)
    res = {
        'coefficient': theta,
        'diagnostic_data': {
            'y': y,
            'd': d,
            'g0': g0,
            'g1': g1,
            'm': m,
        },
        'params': {'score': 'ATE'}
    }
    rep = run_score_diagnostics(res=res, return_summary=True)
    assert 'influence_diagnostics' in rep
    assert 'orthogonality_derivatives' in rep
    assert 'summary' in rep
    assert isinstance(rep['summary'], pd.DataFrame)
