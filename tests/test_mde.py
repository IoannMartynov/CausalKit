"""
Tests for the calculate_mde function in the rct_design module.
"""

import pytest
from causalkit.eda.rct_design import calculate_mde


def test_mde_calculation_for_conversion_data():
    """Test MDE calculation for conversion data."""
    result = calculate_mde(
        sample_size=1000,
        baseline_rate=0.1,
        data_type='conversion'
    )
    
    # Check that the result contains all expected keys
    assert 'mde' in result
    assert 'mde_relative' in result
    assert 'parameters' in result
    
    # Check that the MDE values are positive floats
    assert isinstance(result['mde'], float)
    assert result['mde'] > 0
    
    assert isinstance(result['mde_relative'], float)
    assert result['mde_relative'] > 0
    
    # Check that the parameters dictionary contains expected keys
    assert 'sample_size' in result['parameters']
    assert 'baseline_rate' in result['parameters']
    assert 'data_type' in result['parameters']
    assert result['parameters']['data_type'] == 'conversion'


def test_mde_calculation_for_continuous_data():
    """Test MDE calculation for continuous data."""
    result = calculate_mde(
        sample_size=(500, 500),
        variance=4,
        baseline_rate=10,  # Optional for continuous data
        data_type='continuous'
    )
    
    # Check that the result contains all expected keys
    assert 'mde' in result
    assert 'mde_relative' in result
    assert 'parameters' in result
    
    # Check that the MDE values are positive floats
    assert isinstance(result['mde'], float)
    assert result['mde'] > 0
    
    assert isinstance(result['mde_relative'], float)
    assert result['mde_relative'] > 0
    
    # Check that the parameters dictionary contains expected keys
    assert 'sample_size' in result['parameters']
    assert 'variance' in result['parameters']
    assert 'data_type' in result['parameters']
    assert result['parameters']['data_type'] == 'continuous'


def test_mde_calculation_with_different_sample_allocation():
    """Test MDE calculation with different sample allocation."""
    result = calculate_mde(
        sample_size=1000,
        baseline_rate=0.1,
        data_type='conversion',
        ratio=0.7  # 70% in control, 30% in treatment
    )
    
    # Check that the result contains all expected keys
    assert 'mde' in result
    assert 'mde_relative' in result
    assert 'parameters' in result
    
    # The ratio parameter affects the calculation but might not be included in the parameters dictionary
    # Just verify that the function runs successfully with a custom ratio


def test_mde_calculation_with_different_alpha_and_power():
    """Test MDE calculation with different alpha and power."""
    result = calculate_mde(
        sample_size=1000,
        baseline_rate=0.1,
        data_type='conversion',
        alpha=0.01,  # More stringent significance level
        power=0.9    # Higher power
    )
    
    # Check that the result contains all expected keys
    assert 'mde' in result
    assert 'mde_relative' in result
    assert 'parameters' in result
    
    # Check that the alpha and power parameters were correctly used
    assert 'alpha' in result['parameters']
    assert result['parameters']['alpha'] == 0.01
    
    assert 'power' in result['parameters']
    assert result['parameters']['power'] == 0.9


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main(["-xvs", __file__])