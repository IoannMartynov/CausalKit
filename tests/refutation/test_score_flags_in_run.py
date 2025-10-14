import numpy as np
import pandas as pd

from causalis.refutation.score.score_validation import run_score_diagnostics


def _make_synth(seed=0, n=180):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    beta0 = np.array([0.5, -0.3, 0.2])
    tau = 1.1
    g0 = X @ beta0
    g1 = g0 + tau
    logits = X @ np.array([0.4, -0.15, 0.08])
    m = 1 / (1 + np.exp(-logits))
    d = rng.binomial(1, m).astype(float)
    y = g0 + d * tau + rng.normal(scale=1.0, size=n)
    # AIPW plug-in theta using true nuisances for stability
    m_clip = np.clip(m, 1e-3, 1-1e-3)
    theta_terms = (g1 - g0) + d * (y - g1) / m_clip - (1 - d) * (y - g0) / (1 - m_clip)
    theta = float(theta_terms.mean())
    return y, d, g0, g1, m, theta


def test_run_score_diagnostics_has_flags_by_default():
    y, d, g0, g1, m, theta = _make_synth(seed=202, n=160)
    rep = run_score_diagnostics(y=y, d=d, g0=g0, g1=g1, m=m, theta=theta, score='ATE', return_summary=True)
    assert 'flags' in rep, 'run_score_diagnostics should include flags by default'
    assert 'overall_flag' in rep
    assert rep['overall_flag'] in {"GREEN","YELLOW","RED","NA"}
    assert 'summary' in rep and isinstance(rep['summary'], pd.DataFrame)
    assert 'flag' in rep['summary'].columns
