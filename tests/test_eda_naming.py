import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from causalkit.data.causaldata import CausalData
from causalkit.data.generators import generate_rct
from causalkit.eda.eda import CausalEDA, PropensityModel


def _make_cd(n=800, seed=123):
    df = generate_rct(n=n, split=0.5, random_state=seed, k=4, add_ancillary=False)
    y = "y"; t = "t"
    xcols = [c for c in df.columns if c not in {y, t, "m", "g0", "g1", "propensity", "mu0", "mu1", "cate"}]
    return CausalData(df=df[[y, t] + xcols], treatment=t, outcome=y, confounders=xcols)


def test_fit_m_returns_pm_with_m_and_alias_equal():
    cd = _make_cd(600, 11)
    eda = CausalEDA(cd, ps_model=LogisticRegression(max_iter=1000), n_splits=3, random_state=0)
    pm = eda.fit_m()

    assert isinstance(pm, PropensityModel)
    assert hasattr(pm, "m") and pm.m.shape[0] == len(cd.df)
    # Back-compat alias
    assert np.allclose(pm.m, pm.propensity_scores)

    # Cache equality
    assert hasattr(eda, "_m")
    assert np.allclose(eda._m, pm.m)


def test_auc_m_matches_deprecated_api_and_plots_run():
    cd = _make_cd(500, 21)
    eda = CausalEDA(cd, ps_model=LogisticRegression(max_iter=1000), n_splits=3, random_state=0)
    pm = eda.fit_m()

    auc_new = eda.auc_m()
    auc_old = eda.confounders_roc_auc()  # deprecated path should warn and match
    assert np.isclose(auc_new, auc_old)

    # Plotters should execute without error
    eda.plot_m_overlap()
    eda.plot_ps_overlap()  # deprecated alias

    # PropensityModel plotting and positivity checks
    pm.plot_m_overlap()
    res_new = pm.positivity_check_m()
    res_old = pm.positivity_check()
    assert set(res_new.keys()) == {"bounds", "share_below", "share_above", "flag"}
    # Basic consistency: deprecated returns same as new
    assert res_new["bounds"] == res_old["bounds"]
