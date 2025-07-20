"""
Tests for the causaldata class.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data import generate_rct_data, generate_obs_data, causaldata


@pytest.fixture
def random_seed():
    """Fixture to provide a consistent random seed for tests."""
    return 42


@pytest.fixture
def custom_dataframe():
    """Fixture to provide a custom DataFrame for testing."""
    return pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(5)],
        'ltv': [100, 200, 150, 300, 250],
        'age': [25, 30, 35, 40, 45],
        'invited_friend': [1, 0, 1, 0, 1],
        'treatment': [1, 0, 1, 0, 1]
    })


def test_causaldata_with_rct_data(random_seed):
    """Test causaldata class with RCT data."""
    # Generate RCT data
    rct_df = generate_rct_data(n_users=1000, split=0.5, random_state=random_seed)
    
    # Create causaldata object
    ck_rct = causaldata(
        df=rct_df,
        target='target',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )
    
    # Verify causaldata object properties
    assert ck_rct.target is not None
    assert ck_rct.target.shape[0] == 1000
    assert set(ck_rct.cofounders.columns) == {'age', 'invited_friend'}
    assert abs(ck_rct.treatment.mean() - 0.5) < 0.05  # Treatment ratio should be close to 0.5


def test_causaldata_with_observational_data(random_seed):
    """Test causaldata class with observational data."""
    # Generate observational data
    obs_df = generate_obs_data(n_users=1000, split=0.1, random_state=random_seed)
    
    # Create causaldata object
    ck_obs = causaldata(
        df=obs_df,
        target=None,  # No target column in observational data
        cofounders=['age', 'income', 'education'],
        treatment='treatment'
    )
    
    # Verify causaldata object properties
    assert ck_obs.target is None
    assert set(ck_obs.cofounders.columns) == {'age', 'income', 'education'}
    assert abs(ck_obs.treatment.mean() - 0.1) < 0.05  # Treatment ratio should be close to 0.1


def test_causaldata_with_custom_data(custom_dataframe):
    """Test causaldata class with custom DataFrame."""
    # Create causaldata object
    ck_custom = causaldata(
        df=custom_dataframe,
        target='ltv',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )
    
    # Verify causaldata object properties
    assert ck_custom.target.tolist() == [100, 200, 150, 300, 250]
    assert set(ck_custom.cofounders.columns) == {'age', 'invited_friend'}
    assert ck_custom.treatment.tolist() == [1, 0, 1, 0, 1]