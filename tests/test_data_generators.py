"""
Tests for the data generator functions in the causalkit.data.generators module.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data.generators import generate_rct_data, generate_obs_data


@pytest.fixture
def random_seed():
    """Fixture to provide a consistent random seed for tests."""
    return 42


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
