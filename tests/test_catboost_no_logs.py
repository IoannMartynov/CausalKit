import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from causalkit.data.causaldata import CausalData


def _make_small_causaldata(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    # treatment assignment depends on Xs
    logits = 0.5 * X1 - 0.25 * X2
    p = 1 / (1 + np.exp(-logits))
    T = rng.binomial(1, p)
    # outcome depends on T and Xs
    Y = 1.5 * T + 0.3 * X1 - 0.1 * X2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({
        'y': Y.astype(float),
        't': T.astype(int),
        'x1': X1.astype(float),
        'x2': X2.astype(float),
    })
    return CausalData(df=df, treatment='t', outcome='y', cofounders=['x1', 'x2'])


@pytest.mark.parametrize("module_path, func_name", [
    ("causalkit.inference.ate.dml_ate", "dml_ate"),
    ("causalkit.inference.att.dml", "dml"),
])
def test_catboost_dml_does_not_write_files(module_path, func_name):
    # Prepare tiny dataset
    ck = _make_small_causaldata(n=200, seed=123)

    # Run inside a temporary working directory to detect any side-effect files
    with tempfile.TemporaryDirectory() as tmp:
        cwd_before = os.getcwd()
        try:
            os.chdir(tmp)
            # Sanity check: no catboost_info at start
            assert not os.path.exists(os.path.join(tmp, 'catboost_info'))

            # Import here to avoid pytest collection import side-effects in different CWD
            mod = __import__(module_path, fromlist=[func_name])
            func = getattr(mod, func_name)

            # Execute
            res = func(ck, n_folds=2, n_rep=1)
            assert 'coefficient' in res

            # Verify CatBoost did not create logging directory
            assert not os.path.exists(os.path.join(tmp, 'catboost_info'))
        finally:
            os.chdir(cwd_before)
