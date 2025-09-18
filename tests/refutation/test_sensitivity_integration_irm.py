import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.ate import dml_ate
from causalkit.inference.att import dml_att
from causalkit.refutation.sensitivity import sensitivity_analysis, get_sensitivity_summary


def _make_cd(n=600, random_state=3, target_type="normal"):
    df = generate_rct(n=n, split=0.5, random_state=random_state, target_type=target_type, k=3, add_ancillary=False)
    y = "y"; t = "t"
    xcols = [c for c in df.columns if c not in {y, t, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, t] + xcols], treatment=t, outcome=y, confounders=xcols)


def test_sensitivity_with_dml_ate_runs_and_returns_string():
    cd = _make_cd(n=400, random_state=11, target_type="normal")
    ml_g = RandomForestRegressor(n_estimators=30, random_state=1)
    ml_m = RandomForestClassifier(n_estimators=30, random_state=1)

    res = dml_ate(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3)
    out = sensitivity_analysis(res, cf_y=0.02, cf_d=0.03, rho=1.0)

    assert isinstance(out, str)
    assert "Sensitivity Analysis" in out
    # Integration: summary should also be retrievable
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert "Bounds with CI" in summ


def test_sensitivity_with_dml_att_runs_and_returns_string():
    cd = _make_cd(n=400, random_state=7, target_type="normal")
    ml_g = RandomForestRegressor(n_estimators=25, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=25, random_state=0)

    res = dml_att(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3)
    out = sensitivity_analysis(res, cf_y=0.01, cf_d=0.04, rho=0.8)

    assert isinstance(out, str)
    assert "Sensitivity Analysis" in out
