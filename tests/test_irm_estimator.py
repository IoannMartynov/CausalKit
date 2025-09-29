import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.estimators import IRM


def make_causal_data(n=1000, target_type="normal", random_state=1):
    df = generate_rct(n=n, split=0.5, random_state=random_state, target_type=target_type, k=3, add_ancillary=False)
    # map to expected columns: outcome y, treatment t, confounders any x*
    y = "y"; d = "d"
    xcols = [c for c in df.columns if c not in {y, d, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    cd = CausalData(df=df[[y, d] + xcols], treatment=d, outcome=y, confounders=xcols)
    return cd


def test_irm_ate_runs_and_shapes():
    cd = make_causal_data(n=800, target_type="normal", random_state=42)
    ml_g = RandomForestRegressor(n_estimators=50, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=50, random_state=42)

    est = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, score="ATE", random_state=123)
    est.fit()

    assert est.coef.shape == (1,)
    assert est.se.shape == (1,)
    assert np.isfinite(est.se[0])
    ci = est.confint(level=0.95)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (1, 2)


def test_irm_atte_runs():
    cd = make_causal_data(n=600, target_type="normal", random_state=7)
    ml_g = RandomForestRegressor(n_estimators=40, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=40, random_state=0)

    est = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, score="ATTE", random_state=1)
    est.fit()
    assert np.isfinite(est.coef[0])


def test_irm_binary_outcome_with_classifier():
    cd = make_causal_data(n=800, target_type="binary", random_state=21)
    ml_g = RandomForestClassifier(n_estimators=60, random_state=21)
    ml_m = RandomForestClassifier(n_estimators=60, random_state=21)

    est = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, score="ATE", random_state=21)
    est.fit()
    assert np.isfinite(est.se[0])


def test_irm_raises_on_non_binary_treatment():
    cd = make_causal_data(n=300, target_type="normal", random_state=3)
    # Modify treatment to be non-binary
    df = cd.df.copy()
    df[cd.treatment.name] = df[cd.treatment.name].replace({1: 2})
    cd_bad = CausalData(df=df, treatment=cd.treatment.name, outcome=cd.target.name, confounders=cd.confounders)

    ml_g = RandomForestRegressor(n_estimators=10, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=10, random_state=0)

    with pytest.raises(ValueError):
        IRM(cd_bad, ml_g=ml_g, ml_m=ml_m, n_folds=2).fit()
