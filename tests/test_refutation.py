"""
Tests for the refutation utilities.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from causalkit.data.causaldata import CausalData
from causalkit.refutation import (
    refute_placebo_outcome,
    refute_placebo_treatment,
    refute_subset,
)
from causalkit.refutation.score.placebo import _is_binary, _generate_random_outcome, _generate_random_treatment


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Generate covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Generate treatment (binary)
    treatment_prob = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    treatment = np.random.binomial(1, treatment_prob, n)
    
    # Generate outcome with treatment effect
    outcome = 2 + 0.5 * x1 + 0.3 * x2 + 1.5 * treatment + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'x1': x1,
        'x2': x2
    })
    
    return CausalData(
        df=df,
        treatment='treatment',
        outcome='outcome',
        confounders=['x1', 'x2']
    )


@pytest.fixture
def mock_inference_fn():
    """Create a mock inference function that returns expected format."""
    def inference_fn(data, **kwargs):
        # Mock function that returns realistic results
        return {
            'coefficient': 1.2,
            'p_value': 0.05,
            'std_error': 0.3,
            'confidence_interval': (0.6, 1.8)
        }
    return inference_fn


def test_refute_placebo_outcome_basic(sample_data, mock_inference_fn):
    """Test basic functionality of refute_placebo_outcome."""
    result = refute_placebo_outcome(mock_inference_fn, sample_data, random_state=42)
    
    assert isinstance(result, dict)
    assert 'theta' in result
    assert 'p_value' in result
    assert isinstance(result['theta'], float)
    assert isinstance(result['p_value'], float)


def test_refute_placebo_treatment_basic(sample_data, mock_inference_fn):
    """Test basic functionality of refute_placebo_treatment."""
    result = refute_placebo_treatment(mock_inference_fn, sample_data, random_state=42)
    
    assert isinstance(result, dict)
    assert 'theta' in result
    assert 'p_value' in result
    assert isinstance(result['theta'], float)
    assert isinstance(result['p_value'], float)


def test_refute_subset_basic(sample_data, mock_inference_fn):
    """Test basic functionality of refute_subset."""
    result = refute_subset(mock_inference_fn, sample_data, fraction=0.8, random_state=42)
    
    assert isinstance(result, dict)
    assert 'theta' in result
    assert 'p_value' in result
    assert isinstance(result['theta'], float)
    assert isinstance(result['p_value'], float)


def test_refute_subset_fraction_validation():
    """Test that refute_subset validates fraction parameter."""
    mock_fn = Mock()
    mock_data = Mock()
    
    # Test invalid fractions
    with pytest.raises(ValueError, match="`fraction` must lie in \\(0, 1\\]"):
        refute_subset(mock_fn, mock_data, fraction=0.0)
    
    with pytest.raises(ValueError, match="`fraction` must lie in \\(0, 1\\]"):
        refute_subset(mock_fn, mock_data, fraction=1.5)
    
    with pytest.raises(ValueError, match="`fraction` must lie in \\(0, 1\\]"):
        refute_subset(mock_fn, mock_data, fraction=-0.1)


def test_placebo_outcome_generates_random_outcome(sample_data):
    """Test that refute_placebo_outcome generates random outcome variables instead of shuffling."""
    # Create a mock inference function that returns the mean of the outcome
    def outcome_mean_fn(data, **kwargs):
        df = data.get_df()
        mean_outcome = df[data._target].mean()
        return {'coefficient': mean_outcome, 'p_value': 0.5}
    
    # Get original outcome mean and std
    original_outcome = sample_data.get_df()[sample_data._target]
    original_mean = original_outcome.mean()
    original_std = original_outcome.std()
    
    # Run refutation multiple times and check if we get similar distribution parameters
    means = []
    for seed in range(10):
        result = refute_placebo_outcome(outcome_mean_fn, sample_data, random_state=seed)
        means.append(result['theta'])
    
    # The mean should be approximately the same (continuous outcome generates from normal distribution fitted to original)
    assert abs(np.mean(means) - original_mean) < 0.5  # Mean should be approximately preserved
    
    # Test with fixed seed - should be reproducible
    result1 = refute_placebo_outcome(outcome_mean_fn, sample_data, random_state=42)
    result2 = refute_placebo_outcome(outcome_mean_fn, sample_data, random_state=42)
    assert result1['theta'] == result2['theta']
    
    # Test that generated values are actually different from original (not shuffled)
    def data_comparison_fn(data, **kwargs):
        df = data.get_df()
        return {'coefficient': df[data._target].iloc[0], 'p_value': 0.5}
    
    original_first_value = sample_data.get_df()[sample_data._target].iloc[0]
    generated_result = refute_placebo_outcome(data_comparison_fn, sample_data, random_state=123)
    # Very unlikely that first generated value matches original first value
    # (this tests that we're generating, not shuffling)
    assert generated_result['theta'] != original_first_value or abs(generated_result['theta'] - original_first_value) > 1e-10


def test_placebo_treatment_generates_random_treatment(sample_data):
    """Test that refute_placebo_treatment generates random binary treatment variables."""
    # Create a mock inference function that returns the treatment rate
    def treatment_rate_fn(data, **kwargs):
        df = data.get_df()
        treatment_rate = df[data._treatment].mean()
        return {'coefficient': treatment_rate, 'p_value': 0.5}
    
    # Get original treatment rate
    original_rate = sample_data.get_df()[sample_data._treatment].mean()
    
    # Run refutation multiple times and check treatment rate is preserved
    rates = []
    for seed in range(10):
        result = refute_placebo_treatment(treatment_rate_fn, sample_data, random_state=seed)
        rates.append(result['theta'])
    
    # Treatment rate should be approximately the same across runs (with some binomial variation)
    mean_rate = np.mean(rates)
    assert abs(mean_rate - original_rate) < 0.05  # Allow for binomial sampling variation
    
    # Test reproducibility with fixed seed
    result1 = refute_placebo_treatment(treatment_rate_fn, sample_data, random_state=42)
    result2 = refute_placebo_treatment(treatment_rate_fn, sample_data, random_state=42)
    assert result1['theta'] == result2['theta']
    
    # Test that generated values are different from original (not shuffled)
    def data_comparison_fn(data, **kwargs):
        df = data.get_df()
        return {'coefficient': sum(df[data._treatment].values), 'p_value': 0.5}  # Sum of treatment values
    
    original_sum = sum(sample_data.get_df()[sample_data._treatment].values)
    generated_result = refute_placebo_treatment(data_comparison_fn, sample_data, random_state=123)
    # Sum should be similar (same proportion) but likely not identical (random generation)
    expected_sum = original_rate * len(sample_data.get_df())
    assert abs(generated_result['theta'] - expected_sum) < len(sample_data.get_df()) * 0.1  # Allow for binomial variation


def test_subset_reduces_sample_size(sample_data):
    """Test that refute_subset uses the correct fraction of data."""
    # Create a mock inference function that returns the sample size
    def sample_size_fn(data, **kwargs):
        df = data.get_df()
        return {'coefficient': len(df), 'p_value': 0.5}
    
    original_size = len(sample_data.get_df())
    
    # Test different fractions
    for fraction in [0.5, 0.8, 1.0]:
        result = refute_subset(sample_size_fn, sample_data, fraction=fraction, random_state=42)
        expected_size = int(np.floor(fraction * original_size))
        assert result['theta'] == expected_size


def test_kwargs_passed_through(sample_data):
    """Test that inference_kwargs are passed through to the inference function."""
    def kwarg_capture_fn(data, test_param=None, **kwargs):
        return {'coefficient': test_param or 0, 'p_value': 0.5}
    
    # Test all three functions pass kwargs
    result1 = refute_placebo_outcome(kwarg_capture_fn, sample_data, test_param=1.5)
    assert result1['theta'] == 1.5
    
    result2 = refute_placebo_treatment(kwarg_capture_fn, sample_data, test_param=2.5)
    assert result2['theta'] == 2.5
    
    result3 = refute_subset(kwarg_capture_fn, sample_data, test_param=3.5)
    assert result3['theta'] == 3.5


def test_original_data_unchanged(sample_data, mock_inference_fn):
    """Test that original data is not modified by refutation functions."""
    original_df = sample_data.get_df().copy()
    
    # Run all refutation functions
    refute_placebo_outcome(mock_inference_fn, sample_data, random_state=42)
    refute_placebo_treatment(mock_inference_fn, sample_data, random_state=42)
    refute_subset(mock_inference_fn, sample_data, random_state=42)
    
    # Check that original data is unchanged
    pd.testing.assert_frame_equal(sample_data.get_df(), original_df)


# ------------------------------------------------------------------
# Tests for helper functions
# ------------------------------------------------------------------
def test_is_binary_function():
    """Test the _is_binary helper function with various data types."""
    # Test binary 0/1 data
    binary_01 = pd.Series([0, 1, 0, 1, 1])
    assert _is_binary(binary_01) == True
    
    # Test single value binary data
    binary_single_0 = pd.Series([0, 0, 0])
    assert _is_binary(binary_single_0) == True
    
    binary_single_1 = pd.Series([1, 1, 1])
    assert _is_binary(binary_single_1) == True
    
    # Test True/False binary data
    binary_bool = pd.Series([True, False, True, False])
    assert _is_binary(binary_bool) == True
    
    # Test continuous data
    continuous = pd.Series([1.2, 3.4, 5.6, 7.8])
    assert _is_binary(continuous) == False
    
    # Test integer data with more than 2 values
    multi_int = pd.Series([1, 2, 3, 4, 5])
    assert _is_binary(multi_int) == False
    
    # Test two distinct non-binary values (should be treated as binary)
    two_values = pd.Series([10, 20, 10, 20, 10])
    assert _is_binary(two_values) == True
    
    # Test with NaN values
    binary_with_nan = pd.Series([0, 1, np.nan, 1, 0])
    assert _is_binary(binary_with_nan) == True


def test_generate_random_outcome_binary():
    """Test random outcome generation for binary data."""
    # Create binary outcome data
    binary_outcome = pd.Series([0, 1, 1, 0, 1, 0, 0, 1])
    rng = np.random.default_rng(42)
    
    generated = _generate_random_outcome(binary_outcome, rng)
    
    # Should generate binary values
    unique_vals = set(generated)
    assert unique_vals.issubset({0, 1})
    
    # Should preserve approximate proportion
    original_rate = binary_outcome.mean()
    generated_rate = generated.mean()
    assert abs(generated_rate - original_rate) < 0.3  # Allow for sampling variation
    
    # Should have same length
    assert len(generated) == len(binary_outcome)


def test_generate_random_outcome_continuous():
    """Test random outcome generation for continuous data."""
    # Create continuous outcome data
    continuous_outcome = pd.Series([1.2, 3.4, 5.6, 7.8, 2.1, 4.3, 6.5, 8.7])
    rng = np.random.default_rng(42)
    
    generated = _generate_random_outcome(continuous_outcome, rng)
    
    # Should generate continuous values (not just 0/1)
    assert not _is_binary(pd.Series(generated))
    
    # Should preserve approximate mean and std
    original_mean = continuous_outcome.mean()
    original_std = continuous_outcome.std()
    generated_mean = generated.mean()
    generated_std = generated.std()
    
    assert abs(generated_mean - original_mean) < 1.0  # Allow for sampling variation
    assert abs(generated_std - original_std) < 1.0    # Allow for sampling variation
    
    # Should have same length
    assert len(generated) == len(continuous_outcome)


def test_generate_random_treatment():
    """Test random treatment generation."""
    # Create treatment data with specific proportion
    treatment = pd.Series([0, 1, 0, 0, 1, 1, 0, 1])  # 50% treatment rate
    rng = np.random.default_rng(42)
    
    generated = _generate_random_treatment(treatment, rng)
    
    # Should generate binary values
    unique_vals = set(generated)
    assert unique_vals.issubset({0, 1})
    
    # Should preserve approximate proportion
    original_rate = treatment.mean()
    generated_rate = generated.mean()
    assert abs(generated_rate - original_rate) < 0.3  # Allow for binomial sampling variation
    
    # Should have same length
    assert len(generated) == len(treatment)


def test_binary_outcome_placebo_test():
    """Test refute_placebo_outcome with binary outcome data."""
    # Create data with binary outcome
    np.random.seed(123)
    n = 500
    
    # Generate covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Generate binary treatment
    treatment_prob = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    treatment = np.random.binomial(1, treatment_prob, n)
    
    # Generate binary outcome
    outcome_prob = 1 / (1 + np.exp(-(0.2 + 0.3 * x1 + 0.2 * x2 + 0.5 * treatment)))
    outcome = np.random.binomial(1, outcome_prob, n)
    
    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'x1': x1,
        'x2': x2
    })
    
    binary_data = CausalData(
        df=df,
        treatment='treatment',
        outcome='outcome',
        confounders=['x1', 'x2']
    )
    
    # Mock inference function
    def mock_inference(data, **kwargs):
        return {'coefficient': 0.5, 'p_value': 0.05}
    
    # Test that it works with binary outcome
    result = refute_placebo_outcome(mock_inference, binary_data, random_state=42)
    assert isinstance(result, dict)
    assert 'theta' in result
    assert 'p_value' in result