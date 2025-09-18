import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.ate import dml_ate
from causalkit.refutation.sensitivity import sensitivity_analysis_set


def _make_cd(n=800, seed=123):
    df = generate_rct(n=n, split=0.5, random_state=seed, k=4, add_ancillary=False)
    y = "y"; t = "t"
    xcols = [c for c in df.columns if c not in {y, t, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, t] + xcols], treatment=t, outcome=y, confounders=xcols)


def _fit_ate(cd):
    ml_g = RandomForestRegressor(n_estimators=20, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=20, random_state=0)
    return dml_ate(cd, ml_g=ml_g, ml_m=ml_m, n_folds=2, score="ATE", confidence_level=0.9)


def test_sensitivity_set_returns_dict_of_dfs_for_irm_model():
    cd = _make_cd(600, 11)
    ate = _fit_ate(cd)
    confs = list(cd.confounders)

    res = sensitivity_analysis_set(ate, benchmarking_set=confs, level=0.9, null_hypothesis=0.0)

    assert isinstance(res, dict)
    assert set(res.keys()) == set(confs)
    for name, df in res.items():
        assert isinstance(df, pd.DataFrame)
        # Row index should contain treatment name 't'
        assert "t" in df.index or cd.treatment.name in df.index
        # Basic columns present
        for col in ["theta", "CI lower", "CI upper"]:
            assert col in df.columns
        # Numeric values
        assert np.isfinite(df["theta"].iloc[0])


def test_sensitivity_set_grouped_and_single_string_for_irm_model():
    cd = _make_cd(500, 7)
    ate = _fit_ate(cd)
    confs = list(cd.confounders)

    # Grouped
    grp = [confs[0:2], confs[2:4]]
    res = sensitivity_analysis_set(ate, benchmarking_set=grp, level=0.95)
    assert isinstance(res, dict)
    assert set(res.keys()) == {tuple(confs[0:2]), tuple(confs[2:4])}
    for k, v in res.items():
        assert isinstance(v, pd.DataFrame)

    # Single string returns a DataFrame
    single = sensitivity_analysis_set(ate, benchmarking_set=confs[0], level=0.95)
    assert isinstance(single, pd.DataFrame)
