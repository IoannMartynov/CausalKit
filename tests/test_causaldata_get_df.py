"""
Tests for the get_df method in causaldata class.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data import causaldata


@pytest.fixture
def random_seed():
    """Fixture to provide a consistent random seed for tests."""
    return 42


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
    return causaldata(
        df=test_dataframe,
        target='target',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )


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