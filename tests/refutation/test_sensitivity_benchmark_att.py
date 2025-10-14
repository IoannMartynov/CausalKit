import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.inference.atte import dml_atte
from causalkit.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_benchmark


def _make_cd(n=400, random_state=7, target_type="normal"):
    df = generate_rct(n=n, split=0.5, random_state=random_state, target_type=target_type, k=3, add_ancillary=False)
    y = "y"; d = "d"
    xcols = [c for c in df.columns if c not in {y, d, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, d] + xcols], treatment=d, outcome=y, confounders=xcols)


def test_sensitivity_benchmark_runs_for_atte():
    cd = _make_cd(n=300, random_state=5, target_type="normal")
    ml_g = RandomForestRegressor(n_estimators=20, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=20, random_state=0)

    res = dml_atte(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3)

    # choose one confounder to drop
    x_name = cd.confounders[0]
    df_bench = sensitivity_benchmark(res, benchmarking_set=[x_name])

    # Basic checks
    assert isinstance(df_bench, pd.DataFrame)
    assert df_bench.shape[0] == 1
    for col in ["cf_y", "cf_d", "rho", "theta_long", "theta_short", "delta"]:
        assert col in df_bench.columns
        assert np.isfinite(df_bench[col].iloc[0]) or np.isnan(df_bench[col].iloc[0]) is False
