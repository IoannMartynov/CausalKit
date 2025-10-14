"""
Tests for the bootstrap_diff_means function in the ATT inference module.
"""

import pytest
import numpy as np
import pandas as pd

from causalis.data import CausalData
from causalis.inference.atte.bootstrap_diff_means import bootstrap_diff_means


@pytest.fixture
def random_seed():
    return 123


@pytest.fixture
def cont_test_data(random_seed):
    np.random.seed(random_seed)
    n = 3000
    control_mean = 5.0
    treatment_effect = 1.5
    sd = 2.0

    treatment = np.random.choice([0, 1], size=n)
    target = np.where(
        treatment == 1,
        np.random.normal(control_mean + treatment_effect, sd, size=n),
        np.random.normal(control_mean, sd, size=n),
    )

    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': target,
        'age': np.random.randint(18, 70, size=n)
    })

    return {
        'df': df,
        'n': n,
        'control_mean': control_mean,
        'treatment_effect': treatment_effect,
    }


@pytest.fixture
def causal_data(cont_test_data):
    return CausalData(
        df=cont_test_data['df'],
        outcome='outcome',
        treatment='treatment',
        confounders=['age']
    )


def test_basic_keys_and_types(causal_data):
    res = bootstrap_diff_means(causal_data, n_simul=2000)
    expected = ['p_value', 'absolute_difference', 'absolute_ci', 'relative_difference', 'relative_ci']
    assert all(k in res for k in expected)
    assert isinstance(res['p_value'], float)
    assert 0 <= res['p_value'] <= 1
    assert isinstance(res['absolute_ci'], tuple) and len(res['absolute_ci']) == 2
    assert isinstance(res['relative_ci'], tuple) and len(res['relative_ci']) == 2


def test_effect_size_and_ci(causal_data, cont_test_data):
    res = bootstrap_diff_means(causal_data, n_simul=3000)
    expected = cont_test_data['treatment_effect']
    actual = res['absolute_difference']
    assert abs(actual - expected) < 0.3

    lo, hi = res['absolute_ci']
    assert lo <= expected <= hi


def test_confidence_levels_change_width(causal_data):
    r90 = bootstrap_diff_means(causal_data, confidence_level=0.90, n_simul=2500)
    r95 = bootstrap_diff_means(causal_data, confidence_level=0.95, n_simul=2500)
    r99 = bootstrap_diff_means(causal_data, confidence_level=0.99, n_simul=2500)

    w90 = r90['absolute_ci'][1] - r90['absolute_ci'][0]
    w95 = r95['absolute_ci'][1] - r95['absolute_ci'][0]
    w99 = r99['absolute_ci'][1] - r99['absolute_ci'][0]

    assert w90 < w95 < w99


def test_errors_non_binary_treatment(cont_test_data):
    df = cont_test_data['df'].copy()
    df['treatment'] = np.random.choice([0, 1, 2], size=cont_test_data['n'])
    ck = CausalData(df=df, outcome='outcome', treatment='treatment', confounders=['age'])
    with pytest.raises(ValueError):
        bootstrap_diff_means(ck, n_simul=1000)


def test_invalid_params(causal_data):
    with pytest.raises(ValueError):
        bootstrap_diff_means(causal_data, confidence_level=1.2)
    with pytest.raises(ValueError):
        bootstrap_diff_means(causal_data, confidence_level=-0.1)
    with pytest.raises(ValueError):
        bootstrap_diff_means(causal_data, n_simul=0)
    with pytest.raises(ValueError):
        bootstrap_diff_means(causal_data, n_simul=-10)
