import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.ate import dml_ate


def make_causal_data(n=1000, target_type="normal", random_state=1):
    df = generate_rct(n=n, split=0.5, random_state=random_state, target_type=target_type, k=3, add_ancillary=False)
    y = "y"; t = "t"
    xcols = [c for c in df.columns if c not in {y, t, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, t] + xcols], treatment=t, outcome=y, confounders=xcols)




def test_dml_ate_s_atte_runs():
    cd = make_causal_data(n=600, target_type="normal", random_state=7)
    ml_g = RandomForestRegressor(n_estimators=40, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=40, random_state=0)

    res = dml_ate(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, score="ATTE", confidence_level=0.9)
    assert np.isfinite(res["coefficient"]) and np.isfinite(res["std_error"]) and np.isfinite(res["p_value"])


essentially_bool = [True, False]

def test_dml_ate_s_binary_outcome_with_classifier():
    cd = make_causal_data(n=700, target_type="binary", random_state=21)
    ml_g = RandomForestClassifier(n_estimators=60, random_state=21)
    ml_m = RandomForestClassifier(n_estimators=60, random_state=21)

    res = dml_ate(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3, score="ATE")
    assert np.isfinite(res["std_error"]) and np.isfinite(res["coefficient"]) 


def test_dml_ate_s_raises_on_non_binary_treatment():
    cd = make_causal_data(n=300, target_type="normal", random_state=3)
    df = cd.df.copy()
    df[cd.treatment.name] = df[cd.treatment.name].replace({1: 2})
    cd_bad = CausalData(df=df, treatment=cd.treatment.name, outcome=cd.target.name, confounders=cd.confounders)

    ml_g = RandomForestRegressor(n_estimators=10, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=10, random_state=0)

    with pytest.raises(ValueError):
        dml_ate(cd_bad, ml_g=ml_g, ml_m=ml_m, n_folds=2)
