import numpy as np
import pandas as pd

from causalkit.eda.eda import CausalEDA, CausalDataLite


def make_toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.integers(0, 3, size=n)  # categorical-like
    # treatment depends on x1 and x2
    logits = 0.5 * x1 + 0.8 * (x2 == 1)
    p = 1 / (1 + np.exp(-logits))
    t = rng.binomial(1, p)
    # outcome depends on treatment and x1
    y = 2.0 * t + 0.5 * x1 + rng.normal(0, 1, size=n)
    df = pd.DataFrame({"t": t, "y": y, "x1": x1, "x2": x2})
    return df


def test_docstrings_present():
    assert CausalEDA.__doc__ and "Exploratory diagnostics" in CausalEDA.__doc__
    # instance methods should have docstrings
    for name in [
        "data_health_check",
        "summaries",
        "fit_propensity",
        "treatment_predictability_auc",
        "positivity_check",
        "plot_ps_overlap",
        "balance_table",
        "love_plot",
        "weight_diagnostics",
        "design_report",
    ]:
        assert getattr(CausalEDA, name).__doc__ is not None


def test_outputs_shapes_and_keys():
    df = make_toy_data()
    data = CausalDataLite(df=df, treatment="t", target="y", confounders=["x1", "x2"])
    eda = CausalEDA(data)


    # health
    health = eda.data_health_check()
    for k in ["constant_columns", "n_duplicates", "n_rows"]:
        assert k in health

    # summaries
    summ = eda.summaries()
    assert set(["treatment_rate", "outcome_by_treatment", "naive_diff"]).issubset(summ.keys())

    # propensity and auc
    ps = eda.fit_propensity()
    assert isinstance(ps, np.ndarray) and ps.shape[0] == df.shape[0]
    auc = eda.treatment_predictability_auc(ps)
    assert isinstance(auc, float)

    # positivity
    pos = eda.positivity_check(ps)
    for k in ["bounds", "share_below", "share_above", "flag"]:
        assert k in pos

    # balance table
    bal = eda.balance_table(ps)
    assert set(["covariate", "SMD_unweighted", "SMD_weighted", "flag_unw", "flag_w"]).issubset(bal.columns)

    # weights diagnostics
    wdiag = eda.weight_diagnostics(ps)
    for k in ["ESS_all", "ESS_treated", "ESS_control", "w_all_quantiles"]:
        assert k in wdiag

    # design report
    report = eda.design_report()
    for k in ["health", "summaries", "treat_auc", "positivity", "balance", "weights"]:
        assert k in report
