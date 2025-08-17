import numpy as np
import pytest

from causalkit.refutation.orthogonality import extract_nuisances, aipw_score_ate, aipw_score_att


class DummyDML:
    def __init__(self, preds):
        self.predictions = preds


def test_extract_nuisances_averages_reps():
    n, r = 5, 3
    # create arrays where mean across reps is easy to verify
    ml_m = np.tile(np.linspace(0.2, 0.8, n).reshape(-1, 1), (1, r))
    ml_g0 = np.tile(np.linspace(1.0, 2.0, n).reshape(-1, 1), (1, r))
    ml_g1 = np.tile(np.linspace(2.0, 3.0, n).reshape(-1, 1), (1, r))
    # perturb reps slightly so averaging is meaningful
    ml_m += np.array([0.0, 0.01, -0.01])
    ml_g0 += np.array([0.0, 0.02, -0.02])
    ml_g1 += np.array([0.0, -0.03, 0.03])

    preds = {
        'ml_m': ml_m,
        'ml_g0': ml_g0,
        'ml_g1': ml_g1,
    }
    dml = DummyDML(preds)

    g, m0, m1 = extract_nuisances(dml)

    # Means across reps should match np.mean(axis=1)
    np.testing.assert_allclose(g, ml_m.mean(axis=1))
    np.testing.assert_allclose(m0, ml_g0.mean(axis=1))
    np.testing.assert_allclose(m1, ml_g1.mean(axis=1))


def test_aipw_scores_match_formulas():
    # simple synthetic data
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0, 1, 0, 1], dtype=float)
    m0 = np.array([0.5, 1.5, 2.5, 3.5])
    m1 = np.array([1.5, 2.5, 3.5, 4.5])
    g = np.array([0.3, 0.6, 0.4, 0.7])
    theta = 0.2

    # ATE EIF direct computation
    g_clip = np.clip(g, 0.01, 1-0.01)
    psi_ate_manual = (m1 - m0) + d*(y - m1)/g_clip - (1-d)*(y - m0)/(1-g_clip) - theta
    psi_ate = aipw_score_ate(y, d, m0, m1, g, theta, eps=0.01)
    np.testing.assert_allclose(psi_ate, psi_ate_manual)

    # ATT EIF direct computation
    p1 = d.mean()
    psi_att_manual = ( d*(y - m0 - theta) + (1-d)*(g_clip/(1-g_clip))*(y - m0) ) / (p1 + 1e-12)
    psi_att = aipw_score_att(y, d, m0, m1, g, theta, p1=p1, eps=0.01)
    np.testing.assert_allclose(psi_att, psi_att_manual)


def test_extract_nuisances_requires_keys():
    dml = DummyDML(preds={'ml_g0': np.array([1.0, 2.0]), 'ml_g1': np.array([1.0, 2.0])})
    with pytest.raises(KeyError):
        extract_nuisances(dml)


def test_extract_nuisances_handles_3d_arrays():
    n = 7
    ml_m = np.linspace(0.1, 0.9, n).reshape(n, 1, 1)
    ml_g0 = np.linspace(1.0, 2.0, n).reshape(n, 1, 1)
    ml_g1 = np.linspace(2.0, 3.0, n).reshape(n, 1, 1)
    preds = {'ml_m': ml_m, 'ml_g0': ml_g0, 'ml_g1': ml_g1}
    dml = DummyDML(preds)

    g, m0, m1 = extract_nuisances(dml)

    assert g.shape == (n,)
    assert m0.shape == (n,)
    assert m1.shape == (n,)
    np.testing.assert_allclose(g, ml_m.reshape(n))
    np.testing.assert_allclose(m0, ml_g0.reshape(n))
    np.testing.assert_allclose(m1, ml_g1.reshape(n))
