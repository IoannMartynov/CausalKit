"""
Tests for the ttest function in the inference module.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data import CausalData
from causalkit.inference.ttest import ttest


@pytest.fixture
def random_seed():
    """Fixture to provide a consistent random seed for tests."""
    return 42


@pytest.fixture
def test_data(random_seed):
    """Fixture to provide test data with a known effect size."""
    np.random.seed(random_seed)
    
    n = 1000
    control_mean = 10.0
    treatment_effect = 2.0
    
    # Create data with treatment effect
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'treatment': np.random.choice([0, 1], size=n),
        'age': np.random.randint(18, 65, size=n),
        'gender': np.random.choice([0, 1], size=n),  # Numeric gender: 0 for 'M', 1 for 'F'
    })
    
    # Generate outcome variable with treatment effect
    df['outcome'] = np.where(
        df['treatment'] == 1,
        np.random.normal(control_mean + treatment_effect, 2.0, size=n),  # Treatment group
        np.random.normal(control_mean, 2.0, size=n)                      # Control group
    )
    
    return {
        'df': df,
        'n': n,
        'control_mean': control_mean,
        'treatment_effect': treatment_effect
    }


@pytest.fixture
def causal_data(test_data):
    """Fixture to provide a causaldata object."""
    return CausalData(
        df=test_data['df'],
        outcome='outcome',
        confounders=['age', 'gender'],
        treatment='treatment'
    )


def test_ttest_basic_functionality(causal_data):
    """Test basic functionality of ttest function."""
    result = ttest(causal_data)
    
    # Check that the result contains all expected keys
    expected_keys = ['p_value', 'absolute_difference', 'absolute_ci', 
                     'relative_difference', 'relative_ci']
    assert all(key in result for key in expected_keys)
    
    # Check that p-value is a float between 0 and 1
    assert isinstance(result['p_value'], float)
    assert 0 <= result['p_value'] <= 1
    
    # Check that confidence intervals are tuples of two floats
    assert isinstance(result['absolute_ci'], tuple)
    assert len(result['absolute_ci']) == 2
    assert all(isinstance(x, float) for x in result['absolute_ci'])
    
    assert isinstance(result['relative_ci'], tuple)
    assert len(result['relative_ci']) == 2
    assert all(isinstance(x, float) for x in result['relative_ci'])


def test_ttest_absolute_difference(causal_data, test_data):
    """Test if absolute difference is close to the expected treatment effect."""
    result = ttest(causal_data)
    
    expected_diff = test_data['treatment_effect']
    actual_diff = result['absolute_difference']
    diff_error = abs(actual_diff - expected_diff)
    
    # Allow for some random variation
    assert diff_error < 0.5


def test_ttest_confidence_intervals(causal_data, test_data):
    """Test if confidence intervals contain the true effect."""
    result = ttest(causal_data)
    
    expected_diff = test_data['treatment_effect']
    lower_bound, upper_bound = result['absolute_ci']
    
    assert lower_bound <= expected_diff <= upper_bound


def test_ttest_relative_difference(causal_data, test_data):
    """Test relative difference calculation."""
    result = ttest(causal_data)
    
    expected_rel_diff = (test_data['treatment_effect'] / test_data['control_mean']) * 100
    actual_rel_diff = result['relative_difference']
    rel_diff_error = abs(actual_rel_diff - expected_rel_diff)
    
    # Allow for some random variation
    assert rel_diff_error < 5


def test_ttest_confidence_levels(causal_data):
    """Test different confidence levels."""
    result_90 = ttest(causal_data, confidence_level=0.90)
    result_95 = ttest(causal_data)  # Default is 0.95
    result_99 = ttest(causal_data, confidence_level=0.99)
    
    # Calculate CI widths
    ci_width_90 = result_90['absolute_ci'][1] - result_90['absolute_ci'][0]
    ci_width_95 = result_95['absolute_ci'][1] - result_95['absolute_ci'][0]
    ci_width_99 = result_99['absolute_ci'][1] - result_99['absolute_ci'][0]
    
    # 90% CI should be narrower than 95% CI, which should be narrower than 99% CI
    assert ci_width_90 < ci_width_95 < ci_width_99


def test_ttest_error_no_treatment(test_data):
    """Test error handling when no treatment is specified."""
    # Create CausalData with required parameters
    ck_no_treatment = CausalData(
        df=test_data['df'],
        outcome='outcome',
        treatment='treatment',
        confounders=['age', 'gender']
    )
    
    # Manually set _treatment to empty list to simulate no treatment
    ck_no_treatment._treatment = []
    
    with pytest.raises(ValueError):
        ttest(ck_no_treatment)


def test_ttest_error_no_target(test_data):
    """Test error handling when no outcome is specified."""
    # Create CausalData with required parameters
    ck_no_target = CausalData(
        df=test_data['df'],
        outcome='outcome',
        treatment='treatment',
        confounders=['age', 'gender']
    )
    
    # Manually set _target to empty list to simulate no outcome
    ck_no_target._target = []
    
    with pytest.raises(ValueError):
        ttest(ck_no_target)


def test_ttest_error_non_binary_treatment(test_data):
    """Test error handling when treatment is not binary."""
    df_multi = test_data['df'].copy()
    df_multi['treatment'] = np.random.choice([0, 1, 2], size=test_data['n'])
    
    ck_multi = CausalData(
        df=df_multi,
        outcome='outcome',
        confounders=['age', 'gender'],
        treatment='treatment'
    )
    
    with pytest.raises(ValueError):
        ttest(ck_multi)