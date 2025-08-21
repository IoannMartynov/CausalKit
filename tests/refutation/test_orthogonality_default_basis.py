import numpy as np
import pandas as pd

from causalkit.data.causaldata import CausalData
from causalkit.refutation.orthogonality import refute_irm_orthogonality


def _dummy_inference(data: CausalData, **kwargs):
    # Build simple, stable predictions for testing
    df = data.get_df()
    y = data.target.values.astype(float)
    d = data.treatment.values.astype(float)
    confs = list(data.confounders)
    X = df[confs].values.astype(float) if len(confs) > 0 else np.zeros((len(df), 0))

    if X.shape[1] > 0:
        lin = X.dot(np.arange(1, X.shape[1] + 1)) / (10.0 * X.shape[1])
    else:
        lin = np.zeros(len(df))

    g = 1.0 / (1.0 + np.exp(-lin))
    g = np.clip(g, 0.2, 0.8)  # keep away from extremes to avoid trimming effects

    base = X.dot(np.repeat(0.1, X.shape[1])) if X.shape[1] > 0 else np.zeros(len(df))
    m0 = base
    m1 = base + 0.5

    # simple difference in means as a coefficient
    if (d == 1).any() and (d == 0).any():
        theta = float(y[d == 1].mean() - y[d == 0].mean())
    else:
        theta = 0.0

    class DummyModel:
        def __init__(self, m0, m1, g):
            self.predictions = {
                'ml_m': np.asarray(g),
                'ml_g0': np.asarray(m0),
                'ml_g1': np.asarray(m1),
            }
            self.score = 'ATE'

    se = float(np.std(y, ddof=1) / np.sqrt(len(y))) if len(y) > 1 else 0.0
    return {
        'model': DummyModel(m0, m1, g),
        'coefficient': theta,
        'std_error': se,
    }


def _make_dataset(n: int = 60, p: int = 4):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.2, -0.2, p)
    logits = X.dot(beta)
    probs = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, probs)
    y = 0.5 * d + X.dot(np.linspace(0.1, -0.1, p)) + rng.normal(scale=0.5, size=n)
    cols = {f"x{i+1}": X[:, i] for i in range(p)}
    df = pd.DataFrame({**cols, 'd': d.astype(float), 'y': y.astype(float)})
    confounders = [f"x{i+1}" for i in range(p)]
    cd = CausalData(df=df, treatment='d', outcome='y', confounders=confounders)
    return cd


def test_default_basis_uses_all_confounders_plus_constant():
    data = _make_dataset(n=60, p=4)
    res = refute_irm_orthogonality(
        _dummy_inference,
        data,
        target='ATE',
        n_folds_oos=3,  # keep it light for the test
    )
    ortho = res['orthogonality_derivatives']['full_sample']
    # number of basis functions equals confounders + 1 (constant)
    assert len(ortho) == len(data.confounders) + 1


def test_explicit_n_basis_funcs_is_respected():
    data = _make_dataset(n=60, p=4)
    res = refute_irm_orthogonality(
        _dummy_inference,
        data,
        target='ATE',
        n_basis_funcs=2,  # constant + 1 covariate
        n_folds_oos=3,
    )
    ortho = res['orthogonality_derivatives']['full_sample']
    assert len(ortho) == 2
