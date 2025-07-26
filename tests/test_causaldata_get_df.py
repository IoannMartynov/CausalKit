"""
Tests for the CausalData.get_df method.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data import CausalData


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
    return CausalData(
        df=test_dataframe,
        target='target',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )


def test_get_df_default(causal_data):
    """Test get_df with default parameters."""
    result_df = causal_data.get_df()
    
    # Should include all columns in the CausalData object (target, treatment, cofounders)
    expected_columns = {'target', 'treatment', 'age', 'invited_friend'}
    assert set(result_df.columns) == expected_columns
    assert result_df.shape[0] == 100
    assert result_df is not causal_data.df  # Should be a copy


def test_get_df_specific_columns(causal_data):
    """Test get_df with specific columns."""
    result_df = causal_data.get_df(
        columns=['age'],
        include_target=False,
        include_cofounders=False,
        include_treatment=False
    )
    
    assert set(result_df.columns) == {'age'}
    assert result_df.shape[0] == 100


def test_get_df_include_target_only(causal_data):
    """Test get_df with include_target=True and others False."""
    result_df = causal_data.get_df(
        include_target=True,
        include_cofounders=False,
        include_treatment=False
    )
    
    assert set(result_df.columns) == {'target'}
    assert result_df.shape[0] == 100


def test_get_df_include_cofounders_only(causal_data):
    """Test get_df with include_cofounders=True and others False."""
    result_df = causal_data.get_df(
        include_target=False,
        include_cofounders=True,
        include_treatment=False
    )
    
    assert set(result_df.columns) == {'age', 'invited_friend'}
    assert result_df.shape[0] == 100


def test_get_df_include_treatment_only(causal_data):
    """Test get_df with include_treatment=True and others False."""
    result_df = causal_data.get_df(
        include_target=False,
        include_cofounders=False,
        include_treatment=True
    )
    
    assert set(result_df.columns) == {'treatment'}
    assert result_df.shape[0] == 100


def test_get_df_combination(causal_data):
    """Test get_df with a combination of parameters."""
    result_df = causal_data.get_df(
        columns=['age'],
        include_target=True,
        include_cofounders=False,
        include_treatment=True
    )
    
    assert set(result_df.columns) == {'age', 'target', 'treatment'}
    assert result_df.shape[0] == 100


def test_get_df_duplicate_columns(causal_data):
    """Test get_df with duplicate columns."""
    result_df = causal_data.get_df(
        columns=['age', 'target'],
        include_target=True,
        include_cofounders=True,
        include_treatment=False
    )
    
    # Each column should appear only once
    assert set(result_df.columns) == {'age', 'target', 'invited_friend'}
    assert result_df.shape[0] == 100
    assert len(result_df.columns) == 3  # No duplicates


def test_get_df_no_columns_no_includes(causal_data):
    """Test get_df with no columns and no includes."""
    result_df = causal_data.get_df(
        columns=None,
        include_target=False,
        include_cofounders=False,
        include_treatment=False
    )
    
    # Should return the entire DataFrame
    expected_columns = {'target', 'treatment', 'age', 'invited_friend'}
    assert set(result_df.columns) == expected_columns
    assert result_df.shape[0] == 100


def test_get_df_error_nonexistent_column(causal_data):
    """Test get_df with a non-existent column."""
    with pytest.raises(ValueError):
        causal_data.get_df(columns=['non_existent_column'])