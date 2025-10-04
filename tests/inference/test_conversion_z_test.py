"""
Tests for the conversion_z_test function in the ATT inference module.
"""

import pytest
import numpy as np
import pandas as pd

from causalkit.data import CausalData
from causalkit.inference.atte.conversion_z_test import conversion_z_test


@pytest.fixture
def random_seed():
    return 42


@pytest.fixture
def conv_test_data(random_seed):
    np.random.seed(random_seed)
    n = 5000
    baseline = 0.10
    uplift = 0.03  # absolute uplift in treatment

    treatment = np.random.choice([0, 1], size=n)
    # control prob = baseline; treatment prob = baseline + uplift
    probs = np.where(treatment == 1, baseline + uplift, baseline)
    target = (np.random.rand(n) < probs).astype(int)

    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': target,
        'age': np.random.randint(18, 70, size=n)
    })

    return {
        'df': df,
        'baseline': baseline,
        'uplift': uplift,
        'n': n,
    }


@pytest.fixture
def causal_data(conv_test_data):
    return CausalData(
        df=conv_test_data['df'],
        outcome='outcome',
        treatment='treatment',
        confounders=['age']
    )


def test_basic_keys_and_types(causal_data):
    res = conversion_z_test(causal_data)
    expected_keys = ['p_value', 'absolute_difference', 'absolute_ci', 'relative_difference', 'relative_ci']
    assert all(k in res for k in expected_keys)
    assert isinstance(res['p_value'], float)
    assert 0 <= res['p_value'] <= 1
    assert isinstance(res['absolute_ci'], tuple) and len(res['absolute_ci']) == 2
    assert isinstance(res['relative_ci'], tuple) and len(res['relative_ci']) == 2


def test_effect_size_and_ci(causal_data, conv_test_data):
    res = conversion_z_test(causal_data)
    expected_diff = conv_test_data['uplift']
    actual_diff = res['absolute_difference']
    # allow sampling noise
    assert abs(actual_diff - expected_diff) < 0.015

    lower, upper = res['absolute_ci']
    assert lower <= expected_diff <= upper


def test_relative_difference(causal_data, conv_test_data):
    res = conversion_z_test(causal_data)
    expected_rel = (conv_test_data['uplift'] / conv_test_data['baseline']) * 100
    assert abs(res['relative_difference'] - expected_rel) < 5


def test_confidence_levels_change_width(causal_data):
    res90 = conversion_z_test(causal_data, confidence_level=0.90)
    res95 = conversion_z_test(causal_data, confidence_level=0.95)
    res99 = conversion_z_test(causal_data, confidence_level=0.99)

    w90 = res90['absolute_ci'][1] - res90['absolute_ci'][0]
    w95 = res95['absolute_ci'][1] - res95['absolute_ci'][0]
    w99 = res99['absolute_ci'][1] - res99['absolute_ci'][0]

    assert w90 < w95 < w99


def test_errors_non_binary_treatment(conv_test_data):
    df = conv_test_data['df'].copy()
    df['treatment'] = np.random.choice([0, 1, 2], size=conv_test_data['n'])
    ck = CausalData(df=df, outcome='outcome', treatment='treatment', confounders=['age'])
    with pytest.raises(ValueError):
        conversion_z_test(ck)


def test_errors_non_binary_target(conv_test_data):
    df = conv_test_data['df'].copy()
    # make the outcome non-binary
    df['outcome'] = np.random.normal(size=conv_test_data['n'])
    ck = CausalData(df=df, outcome='outcome', treatment='treatment', confounders=['age'])
    with pytest.raises(ValueError):
        conversion_z_test(ck)
