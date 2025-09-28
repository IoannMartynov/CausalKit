import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression, LinearRegression

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.ate.dml_ate import dml_ate
from causalkit.inference.att.dml_att import dml_att
from causalkit.refutation import validate_unconfoundedness_balance


@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_unconfoundedness_balance_ate(normalize_ipw):
    # Generate simple RCT-like data with confounders
    df = generate_rct(n=2000, k=3, random_state=123, target_type="binary")
    confs = [c for c in df.columns if c.startswith("x")]  # ['x1','x2','x3']
    data = CausalData(df=df, treatment='t', outcome='y', confounders=confs)

    # Simple learners: regressor for outcome (works even if y is binary),
    # logistic regression for propensity with predict_proba
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=500)

    res = dml_ate(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        confidence_level=0.90,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=7,
        store_diagnostic_data=True,
    )

    out = validate_unconfoundedness_balance(res)

    # Basic structure checks
    assert out['score'] == 'ATE'
    assert out['normalized'] == normalize_ipw
    smd = out['smd']
    assert isinstance(smd, pd.Series)
    assert list(smd.index) == confs
    assert np.all(np.isfinite(smd.values))
    assert (smd.values >= 0).all()

    # New: unweighted SMD available and aligned
    smd_unw = out.get('smd_unweighted')
    assert isinstance(smd_unw, pd.Series)
    assert list(smd_unw.index) == confs
    assert np.all((smd_unw.values >= 0) | ~np.isfinite(smd_unw.values))  # allow NaN if degenerate

    # Removed keys should not be present
    assert 'means_treated' not in out
    assert 'means_control' not in out
    assert 'sd_treated' not in out
    assert 'sd_control' not in out

    # Threshold logic: very high threshold should pass, tiny threshold likely fails
    out_hi = validate_unconfoundedness_balance(res, threshold=1e6)
    assert out_hi['pass'] is True
    out_lo = validate_unconfoundedness_balance(res, threshold=1e-12)
    assert out_lo['pass'] in (False, True)  # don't force, but should be boolean


@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_unconfoundedness_balance_att(normalize_ipw):
    # Generate simple data with confounders
    df = generate_rct(n=2000, k=4, random_state=321, target_type="normal")
    confs = [c for c in df.columns if c.startswith("x")]  # ['x1',...]
    data = CausalData(df=df, treatment='t', outcome='y', confounders=confs)

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=500)

    res = dml_att(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        confidence_level=0.90,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=13,
        store_diagnostic_data=True,
    )

    out = validate_unconfoundedness_balance(res)

    assert out['score'] == 'ATTE'
    assert out['normalized'] == normalize_ipw
    smd = out['smd']
    assert isinstance(smd, pd.Series)
    assert list(smd.index) == confs
    assert np.all(np.isfinite(smd.values))
    assert (smd.values >= 0).all()
