"""
Tests for the data generator functions and causaldata get_df method.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data.generators import generate_rct_data, generate_obs_data
from causalkit.data import CausalData


@pytest.fixture
def random_seed():
    """Fixture to provide a consistent random seed for tests."""
    return 42


# Fixtures for data generators tests
# No additional fixtures needed

# Fixtures for causaldata get_df tests
@pytest.fixture
def test_dataframe(random_seed):
    """Fixture to provide a test DataFrame."""
    np.random.seed(random_seed)
    
    return pd.DataFrame({
        'user_id': range(1, 101),
        'treatment': np.random.choice([0, 1], size=100),
        'age': np.random.randint(18, 65, size=100),
        'gender': np.random.choice(['M', 'F'], size=100),
        'invited_friend': np.random.choice([0, 1], size=100),
        'target': np.random.normal(0, 1, size=100)
    })


@pytest.fixture
def causal_data(test_dataframe):
    """Fixture to provide a causaldata object."""
    return CausalData(
        df=test_dataframe,
        target='target',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )


# Tests for data generators
def test_generate_rct_data_default_parameters(random_seed):
    """Test generate_rct_data with default parameters."""
    rct_df = generate_rct_data(random_state=random_seed)
    
    # Check basic properties
    assert rct_df.shape[0] == 20000  # Default n_users
    
    # Check required columns exist
    required_columns = {'user_id', 'treatment', 'target', 'age', 'invited_friend'}
    assert required_columns.issubset(set(rct_df.columns))
    
    # Check treatment ratio
    treatment_ratio = rct_df['treatment'].mean()
    assert abs(treatment_ratio - 0.5) < 0.01  # Default split is 0.5
    
    # Check user_id format (should be UUID)
    assert all(isinstance(uid, str) and len(uid) == 36 for uid in rct_df['user_id'].head())


def test_generate_rct_data_binary_target(random_seed):
    """Test generate_rct_data with custom parameters and binary target."""
    rct_binary_df = generate_rct_data(
        n_users=1000,
        split=0.3,
        target_type="binary",
        target_params={"p": {"A": 0.15, "B": 0.20}},
        random_state=random_seed
    )
    
    treatment_ratio = rct_binary_df['treatment'].mean()
    control_conversion = rct_binary_df[rct_binary_df['treatment'] == 0]['target'].mean()
    treatment_conversion = rct_binary_df[rct_binary_df['treatment'] == 1]['target'].mean()
    
    assert rct_binary_df.shape[0] == 1000
    assert abs(treatment_ratio - 0.3) < 0.01
    assert abs(control_conversion - 0.15) < 0.02
    assert abs(treatment_conversion - 0.20) < 0.02


def test_generate_rct_data_normal_target(random_seed):
    """Test generate_rct_data with normal target."""
    rct_normal_df = generate_rct_data(
        n_users=1000,
        split=0.5,
        target_type="normal",
        target_params={"mean": {"A": 10.0, "B": 12.0}, "std": 2.0},
        random_state=random_seed
    )
    
    control_mean = rct_normal_df[rct_normal_df['treatment'] == 0]['target'].mean()
    treatment_mean = rct_normal_df[rct_normal_df['treatment'] == 1]['target'].mean()
    control_std = rct_normal_df[rct_normal_df['treatment'] == 0]['target'].std()
    treatment_std = rct_normal_df[rct_normal_df['treatment'] == 1]['target'].std()
    
    assert abs(control_mean - 10.0) < 0.2
    assert abs(treatment_mean - 12.0) < 0.2
    assert abs(control_std - 2.0) < 0.2
    assert abs(treatment_std - 2.0) < 0.2


def test_generate_rct_data_nonnormal_target(random_seed):
    """Test generate_rct_data with non-normal target."""
    rct_nonnormal_df = generate_rct_data(
        n_users=1000,
        split=0.5,
        target_type="nonnormal",
        target_params={"shape": 3.0, "scale": {"A": 1.0, "B": 1.5}},
        random_state=random_seed
    )
    
    control_mean = rct_nonnormal_df[rct_nonnormal_df['treatment'] == 0]['target'].mean()
    treatment_mean = rct_nonnormal_df[rct_nonnormal_df['treatment'] == 1]['target'].mean()
    
    # Expected means for gamma distribution: shape * scale
    expected_control_mean = 3.0 * 1.0
    expected_treatment_mean = 3.0 * 1.5
    
    assert abs(control_mean - expected_control_mean) < 0.3
    assert abs(treatment_mean - expected_treatment_mean) < 0.3


def test_generate_rct_data_invalid_target_type(random_seed):
    """Test generate_rct_data with invalid target_type."""
    with pytest.raises(ValueError):
        generate_rct_data(target_type="invalid", random_state=random_seed)


def test_generate_obs_data_default_parameters(random_seed):
    """Test generate_obs_data with default parameters."""
    obs_df = generate_obs_data(random_state=random_seed)
    
    assert obs_df.shape[0] == 20000  # Default n_users
    assert set(obs_df.columns) >= {'user_id', 'treatment', 'age', 'income', 'education', 'region'}
    
    # Check treatment ratio
    treatment_ratio = obs_df['treatment'].mean()
    assert abs(treatment_ratio - 0.1) < 0.01  # Default split is 0.1
    
    # Check user_id format (should be UUID)
    assert all(isinstance(uid, str) and len(uid) == 36 for uid in obs_df['user_id'].head())


def test_generate_obs_data_custom_parameters():
    """Test generate_obs_data with custom parameters."""
    obs_custom_df = generate_obs_data(n_users=1000, split=0.2, random_state=123)
    
    treatment_ratio = obs_custom_df['treatment'].mean()
    assert obs_custom_df.shape[0] == 1000
    assert abs(treatment_ratio - 0.2) < 0.01


def test_generate_obs_data_covariate_distributions(random_seed):
    """Test covariate distributions in generate_obs_data."""
    obs_check_df = generate_obs_data(n_users=5000, split=0.1, random_state=random_seed)
    
    # Check if education levels are distributed as expected
    education_counts = obs_check_df['education'].value_counts(normalize=True)
    expected_education = {
        'high_school': 0.3,
        'bachelor': 0.4,
        'master': 0.2,
        'phd': 0.1
    }
    
    for edu_level, expected_prop in expected_education.items():
        assert abs(education_counts[edu_level] - expected_prop) < 0.02
    
    # Check if regions are distributed evenly (5 regions)
    region_counts = obs_check_df['region'].value_counts(normalize=True)
    for region_prop in region_counts.values:
        assert abs(region_prop - 0.2) < 0.02


# Tests for causaldata get_df method
def test_get_entire_dataframe(causal_data, test_dataframe):
    """Test getting the entire DataFrame."""
    result_df = causal_data.get_df()
    
    assert result_df.shape == test_dataframe.shape
    assert set(result_df.columns) == set(test_dataframe.columns)
    assert result_df is not test_dataframe  # Should be a copy


def test_get_specific_columns(causal_data):
    """Test getting specific columns."""
    result_df = causal_data.get_df(columns=['user_id', 'gender'])
    
    assert result_df.shape[0] == 100  # Number of rows should be the same
    assert set(result_df.columns) == {'user_id', 'gender'}


def test_get_target_columns(causal_data):
    """Test getting target columns."""
    result_df = causal_data.get_df(include_target=True)
    
    assert result_df.shape[0] == 100
    assert 'target' in result_df.columns


def test_get_cofounder_columns(causal_data):
    """Test getting cofounder columns."""
    result_df = causal_data.get_df(include_cofounders=True)
    
    assert result_df.shape[0] == 100
    assert set(['age', 'invited_friend']).issubset(set(result_df.columns))


def test_get_treatment_columns(causal_data):
    """Test getting treatment columns."""
    result_df = causal_data.get_df(include_treatment=True)
    
    assert result_df.shape[0] == 100
    assert 'treatment' in result_df.columns


def test_get_combination_of_columns(causal_data):
    """Test getting a combination of columns."""
    result_df = causal_data.get_df(
        columns=['user_id', 'gender'],
        include_target=True,
        include_treatment=True
    )
    
    assert result_df.shape[0] == 100
    assert set(result_df.columns) == {'user_id', 'gender', 'target', 'treatment'}


def test_handle_duplicate_columns(causal_data):
    """Test handling of duplicate columns."""
    result_df = causal_data.get_df(
        columns=['age', 'gender'],
        include_cofounders=True  # 'age' is also in cofounders
    )
    
    assert result_df.shape[0] == 100
    # Each column should appear only once
    assert set(result_df.columns) == {'age', 'gender', 'invited_friend'}


def test_error_handling_for_non_existent_columns(causal_data):
    """Test error handling for non-existent columns."""
    with pytest.raises(ValueError):
        causal_data.get_df(columns=['non_existent_column'])