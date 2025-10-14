from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalis.data.causaldata import CausalData
from causalis.data.generators import generate_rct
from causalis.inference.ate import dml_ate
from causalis.inference.atte import dml_atte
from causalis.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_analysis, get_sensitivity_summary


def _make_cd(n=600, random_state=3, target_type="normal"):
    df = generate_rct(n=n, split=0.5, random_state=random_state, target_type=target_type, k=3, add_ancillary=False)
    y = "y"; d = "d"
    xcols = [c for c in df.columns if c not in {y, d, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, d] + xcols], treatment=d, outcome=y, confounders=xcols)


def test_sensitivity_with_dml_ate_runs_and_returns_dict():
    cd = _make_cd(n=400, random_state=11, target_type="normal")
    ml_g = RandomForestRegressor(n_estimators=30, random_state=1)
    ml_m = RandomForestClassifier(n_estimators=30, random_state=1)

    res = dml_ate(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3)
    out = sensitivity_analysis(res, cf_y=0.02, cf_d=0.03, rho=1.0)

    assert isinstance(out, dict)
    # Integration: summary should be retrievable via the getter
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert any(kw in summ for kw in ("Bias-aware Interval", "Intervals"))


def test_sensitivity_with_dml_att_runs_and_returns_dict():
    cd = _make_cd(n=400, random_state=7, target_type="normal")
    ml_g = RandomForestRegressor(n_estimators=25, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=25, random_state=0)

    res = dml_atte(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3)
    out = sensitivity_analysis(res, cf_y=0.01, cf_d=0.04, rho=0.8)

    assert isinstance(out, dict)
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert "Bias-aware Interval" in summ
