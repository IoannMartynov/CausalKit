import numpy as np
import pandas as pd

from causalkit.data.causaldata import CausalData
from causalkit.eda import CausalEDA


def make_synth(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, size=n)
    invited_friend = rng.integers(0, 2, size=n)
    # treatment depends on covariates
    logits = -2.0 + 0.04 * age + 0.8 * invited_friend
    p = 1 / (1 + np.exp(-logits))
    t = (rng.random(n) < p).astype(int)
    # outcome depends on t and covariates
    y = 0.5 * t + 0.02 * age + 0.3 * invited_friend + rng.normal(0, 1, size=n)
    df = pd.DataFrame({
        'target': y,
        'treatment': t,
        'age': age,
        'invited_friend': invited_friend.astype(float),  # keep numeric to satisfy CausalData
    })
    return df


def test_design_report_and_core_metrics():
    df = make_synth()
    cd = CausalData(df=df, treatment='treatment', target='target', cofounders=['age', 'invited_friend'])

    eda = CausalEDA(cd)
    report = eda.design_report()

    # basic keys present
    assert set(['health', 'missing', 'summaries', 'treat_auc', 'positivity', 'balance', 'weights']).issubset(set(report.keys()))

    # AUC is between 0.5 and 1 (should be > 0.5 given signal)
    auc = report['treat_auc']
    assert 0.5 <= auc <= 1.0

    # positivity dict fields
    pos = report['positivity']
    assert 'share_below' in pos and 'share_above' in pos and 'bounds' in pos

    # balance table columns
    bal = report['balance']
    for col in ['covariate', 'SMD_unweighted', 'SMD_weighted', 'flag_unw', 'flag_w']:
        assert col in bal.columns

    # weights diagnostics
    wdiag = report['weights']
    assert 'ESS_all' in wdiag and 'ESS_treated' in wdiag and 'ESS_control' in wdiag and 'w_all_quantiles' in wdiag


def test_fit_propensity_and_balance_table_direct_calls():
    df = make_synth(seed=42)
    cd = CausalData(df=df, treatment='treatment', target='target', cofounders=['age', 'invited_friend'])

    eda = CausalEDA(cd)
    ps = eda.fit_propensity()
    assert ps.shape[0] == df.shape[0]
    assert np.all(ps > 0) and np.all(ps < 1)

    auc = eda.treatment_predictability_auc(ps)
    assert 0.5 <= auc <= 1.0

    bal = eda.balance_table(ps)
    assert not bal.empty
