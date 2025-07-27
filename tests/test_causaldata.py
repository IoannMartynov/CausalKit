"""
Tests for the CausalData class.
"""

import pytest
import pandas as pd
import numpy as np
from causalkit.data import generate_rct_data, generate_obs_data, CausalData


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


# Tests from test_ckit.py
def test_causaldata_with_rct_data(random_seed):
    """Test causaldata class with RCT data."""
    # Generate RCT data
    rct_df = generate_rct_data(n_users=1000, split=0.5, random_state=random_seed)
    
    # Create causaldata object
    ck_rct = CausalData(
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
    ck_obs = CausalData(
        df=obs_df,
        target='income',  # Use 'income' as target column
        cofounders=['age'],  # Use only numeric columns as cofounders
        treatment='treatment'
    )
    
    # Verify causaldata object properties
    assert ck_obs.target is not None
    assert set(ck_obs.cofounders.columns) == {'age'}
    assert abs(ck_obs.treatment.mean() - 0.1) < 0.05  # Treatment ratio should be close to 0.1


def test_causaldata_with_custom_data(custom_dataframe):
    """Test causaldata class with custom DataFrame."""
    # Create causaldata object
    ck_custom = CausalData(
        df=custom_dataframe,
        target='ltv',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )
    
    # Verify causaldata object properties
    assert ck_custom.target.tolist() == [100, 200, 150, 300, 250]
    assert set(ck_custom.cofounders.columns) == {'age', 'invited_friend'}
    assert ck_custom.treatment.tolist() == [1, 0, 1, 0, 1]


def test_causaldata_missing_target(custom_dataframe):
    """Test that an error is raised when target is not provided."""
    with pytest.raises(TypeError):
        CausalData(
            df=custom_dataframe,
            cofounders=['age', 'invited_friend'],
            treatment='treatment'
        )


def test_causaldata_missing_treatment(custom_dataframe):
    """Test that an error is raised when treatment is not provided."""
    with pytest.raises(TypeError):
        CausalData(
            df=custom_dataframe,
            target='ltv',
            cofounders=['age', 'invited_friend']
        )


def test_causaldata_with_nan_values():
    """Test that an error is raised when the DataFrame contains NaN values."""
    # Create a DataFrame with NaN values
    df_with_nan = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(5)],
        'ltv': [100, np.nan, 150, 300, 250],
        'age': [25, 30, 35, 40, 45],
        'invited_friend': [1, 0, 1, 0, 1],
        'treatment': [1, 0, 1, 0, 1]
    })
    
    with pytest.raises(ValueError) as excinfo:
        CausalData(
            df=df_with_nan,
            target='ltv',
            cofounders=['age', 'invited_friend'],
            treatment='treatment'
        )
    
    assert "DataFrame contains NaN values" in str(excinfo.value)


def test_causaldata_with_non_numeric_target():
    """Test that an error is raised when target column is not numeric."""
    # Create a DataFrame with non-numeric target column
    df_with_non_numeric_target = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(5)],
        'target': ['high', 'low', 'medium', 'high', 'low'],  # Non-numeric target
        'age': [25, 30, 35, 40, 45],
        'invited_friend': [1, 0, 1, 0, 1],
        'treatment': [1, 0, 1, 0, 1]
    })
    
    with pytest.raises(ValueError) as excinfo:
        CausalData(
            df=df_with_non_numeric_target,
            target='target',
            cofounders=['age', 'invited_friend'],
            treatment='treatment'
        )
    
    assert "must contain only int or float values" in str(excinfo.value)
    assert "target" in str(excinfo.value)


def test_causaldata_with_non_numeric_cofounders():
    """Test that an error is raised when cofounders column is not numeric."""
    # Create a DataFrame with non-numeric cofounders column
    df_with_non_numeric_cofounders = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(5)],
        'target': [100, 200, 150, 300, 250],
        'age': [25, 30, 35, 40, 45],
        'category': ['A', 'B', 'C', 'A', 'B'],  # Non-numeric cofounder
        'treatment': [1, 0, 1, 0, 1]
    })
    
    with pytest.raises(ValueError) as excinfo:
        CausalData(
            df=df_with_non_numeric_cofounders,
            target='target',
            cofounders=['age', 'category'],
            treatment='treatment'
        )
    
    assert "must contain only int or float values" in str(excinfo.value)
    assert "cofounders" in str(excinfo.value)


def test_causaldata_with_non_numeric_treatment():
    """Test that an error is raised when treatment column is not numeric."""
    # Create a DataFrame with non-numeric treatment column
    df_with_non_numeric_treatment = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(5)],
        'target': [100, 200, 150, 300, 250],
        'age': [25, 30, 35, 40, 45],
        'invited_friend': [1, 0, 1, 0, 1],
        'treatment': ['A', 'B', 'A', 'B', 'A']  # Non-numeric treatment
    })
    
    with pytest.raises(ValueError) as excinfo:
        CausalData(
            df=df_with_non_numeric_treatment,
            target='target',
            cofounders=['age', 'invited_friend'],
            treatment='treatment'
        )
    
    assert "must contain only int or float values" in str(excinfo.value)
    assert "treatment" in str(excinfo.value)


def test_causaldata_truncated_dataframe(custom_dataframe):
    """Test that CausalData stores only the relevant columns (target, treatment, cofounders)."""
    # Create causaldata object
    ck = CausalData(
        df=custom_dataframe,
        target='ltv',
        cofounders=['age', 'invited_friend'],
        treatment='treatment'
    )
    
    # Verify that only the relevant columns are stored
    assert set(ck.df.columns) == {'ltv', 'age', 'invited_friend', 'treatment'}
    assert 'user_id' not in ck.df.columns
    
    # Verify that the stored data is correct
    assert ck.target.tolist() == [100, 200, 150, 300, 250]
    assert set(ck.cofounders.columns) == {'age', 'invited_friend'}
    assert ck.treatment.tolist() == [1, 0, 1, 0, 1]
    
    # Test get_df with all columns
    df_all = ck.get_df()
    assert set(df_all.columns) == {'ltv', 'age', 'invited_friend', 'treatment'}
    assert df_all.shape[1] == 4
    
    # Test __repr__ format
    repr_str = repr(ck)
    assert "df=(5, 4)" in repr_str
    assert "target='ltv'" in repr_str
    assert "cofounders=['age', 'invited_friend']" in repr_str
    assert "treatment='treatment'" in repr_str


# Tests from test_causaldata_get_df.py
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