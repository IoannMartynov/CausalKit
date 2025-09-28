"""
Tests for the sensitivity analysis utilities.
"""

import pytest
from unittest.mock import Mock

from causalkit.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_analysis, get_sensitivity_summary


@pytest.fixture
def mock_dml_model():
    """Create a mock DoubleML model with sensitivity analysis capabilities."""
    model = Mock()
    model.sensitivity_analysis = Mock()
    
    # Create a mock sensitivity summary string (as DoubleML actually returns)
    summary_string = """================== Sensitivity Analysis ==================

------------------ Scenario          ------------------
Significance Level: level=0.95
Sensitivity parameters: cf_y=0.04; cf_d=0.03, rho=1.0

------------------ Bounds with CI    ------------------
           CI lower  theta lower     theta  theta upper  CI upper
e401     1306.608818   3593.56223  7710.567004  11935.392833  14114.64269

------------------ Robustness Values ------------------
           H_0    RV (%)  RVa (%)
e401       0.0  6.240295  4.339269"""
    model.sensitivity_summary = summary_string
    
    return model


@pytest.fixture
def valid_effect_estimation(mock_dml_model):
    """Create a valid effect estimation dictionary."""
    return {
        'coefficient': 7710.567004,
        'std_error': 1500.0,
        'p_value': 0.001,
        'confidence_interval': (4710.567004, 10710.567004),
        'model': mock_dml_model
    }


def test_sensitivity_analysis_basic_functionality(valid_effect_estimation):
    """Test basic sensitivity analysis functionality."""
    result = sensitivity_analysis(
        valid_effect_estimation, 
        cf_y=0.04, 
        cf_d=0.03, 
        rho=1.0, 
        level=0.95
    )
    
    # Check that result is a string
    assert isinstance(result, str)
    
    # Check for expected content in the output
    assert "================== Sensitivity Analysis ==================" in result
    assert "Significance Level: level=0.95" in result
    assert "Sensitivity parameters: cf_y=0.04; cf_d=0.03, rho=1.0" in result
    assert "------------------ Bounds with CI    ------------------" in result
    assert "------------------ Robustness Values ------------------" in result
    assert "e401" in result
    
    # Verify that sensitivity_analysis was called on the model
    valid_effect_estimation['model'].sensitivity_analysis.assert_called_once_with(
        cf_y=0.04, cf_d=0.03, rho=1.0
    )


def test_sensitivity_analysis_different_parameters(valid_effect_estimation):
    """Test sensitivity analysis with different parameters."""
    result = sensitivity_analysis(
        valid_effect_estimation,
        cf_y=0.02,
        cf_d=0.05,
        rho=0.8,
        level=0.90
    )
    
    # The mock returns a hardcoded string, so we focus on testing the function call
    # and that a string is returned
    assert isinstance(result, str)
    assert "Sensitivity Analysis" in result
    
    # Verify correct parameters were passed to model
    valid_effect_estimation['model'].sensitivity_analysis.assert_called_with(
        cf_y=0.02, cf_d=0.05, rho=0.8
    )


def test_sensitivity_analysis_missing_model_key():
    """Test that ValueError is raised when 'model' key is missing."""
    invalid_estimation = {
        'coefficient': 5.0,
        'p_value': 0.05
    }
    
    with pytest.raises(ValueError, match="effect_estimation must contain a 'model' key"):
        sensitivity_analysis(invalid_estimation, cf_y=0.04, cf_d=0.03)


def test_sensitivity_analysis_invalid_input_type():
    """Test that TypeError is raised for invalid input type."""
    with pytest.raises(TypeError, match="effect_estimation must be a dictionary"):
        sensitivity_analysis("not a dict", cf_y=0.04, cf_d=0.03)


def test_sensitivity_analysis_model_without_sensitivity_support():
    """Test that TypeError is raised when model doesn't support sensitivity analysis."""
    model_without_sensitivity = Mock()
    # Remove sensitivity_analysis method
    del model_without_sensitivity.sensitivity_analysis
    
    invalid_estimation = {
        'coefficient': 5.0,
        'model': model_without_sensitivity
    }
    
    with pytest.raises(TypeError, match="must be a DoubleML object that supports sensitivity analysis"):
        sensitivity_analysis(invalid_estimation, cf_y=0.04, cf_d=0.03)


def test_sensitivity_analysis_model_error(valid_effect_estimation):
    """Test RuntimeError when model.sensitivity_analysis fails."""
    valid_effect_estimation['model'].sensitivity_analysis.side_effect = Exception("Model error")
    
    with pytest.raises(RuntimeError, match="Failed to perform sensitivity analysis"):
        sensitivity_analysis(valid_effect_estimation, cf_y=0.04, cf_d=0.03)


def test_sensitivity_analysis_no_summary_generated(valid_effect_estimation):
    """Test RuntimeError when no sensitivity summary is generated."""
    # Remove sensitivity_summary attribute after sensitivity_analysis is called
    def remove_summary(*args, **kwargs):
        if hasattr(valid_effect_estimation['model'], 'sensitivity_summary'):
            del valid_effect_estimation['model'].sensitivity_summary
    
    valid_effect_estimation['model'].sensitivity_analysis.side_effect = remove_summary
    
    with pytest.raises(RuntimeError, match="Sensitivity analysis did not generate a summary"):
        sensitivity_analysis(valid_effect_estimation, cf_y=0.04, cf_d=0.03)


def test_get_sensitivity_summary_valid_model(valid_effect_estimation):
    """Test getting sensitivity summary from a valid model."""
    summary = get_sensitivity_summary(valid_effect_estimation)
    
    assert isinstance(summary, str)
    assert "e401" in summary
    assert "Sensitivity Analysis" in summary
    assert "Bounds with CI" in summary


def test_get_sensitivity_summary_missing_model():
    """Test get_sensitivity_summary with missing model key."""
    invalid_estimation = {'coefficient': 5.0}
    
    result = get_sensitivity_summary(invalid_estimation)
    assert result is None


def test_get_sensitivity_summary_no_summary_attribute():
    """Test get_sensitivity_summary when model has no sensitivity_summary."""
    model_no_summary = Mock()
    del model_no_summary.sensitivity_summary  # Remove the attribute
    
    estimation = {'model': model_no_summary}
    result = get_sensitivity_summary(estimation)
    assert result is None


def test_sensitivity_analysis_output_format(valid_effect_estimation):
    """Test that the output format matches the expected structure."""
    result = sensitivity_analysis(valid_effect_estimation, cf_y=0.04, cf_d=0.03)
    lines = result.split('\n')
    
    # Check structure
    assert lines[0] == "================== Sensitivity Analysis =================="
    assert lines[1] == ""
    assert lines[2] == "------------------ Scenario          ------------------"
    assert "Significance Level: level=0.95" in lines[3]
    assert "Sensitivity parameters: cf_y=0.04; cf_d=0.03, rho=1.0" in lines[4]
    assert lines[5] == ""
    assert lines[6] == "------------------ Bounds with CI    ------------------"
    
    # Check that there are headers and data rows
    bounds_section_start = 6
    robustness_section_idx = None
    for i, line in enumerate(lines):
        if "------------------ Robustness Values ------------------" in line:
            robustness_section_idx = i
            break
    
    assert robustness_section_idx is not None
    
    # Check that there's data between bounds header and robustness section
    assert robustness_section_idx > bounds_section_start + 2


def test_sensitivity_analysis_with_fallback_columns():
    """Test sensitivity analysis with different output format (but still string)."""
    model = Mock()
    model.sensitivity_analysis = Mock()
    
    # Create summary string with slightly different format
    summary_string = """================== Sensitivity Analysis ==================

------------------ Scenario          ------------------
Significance Level: level=0.95
Sensitivity parameters: cf_y=0.04; cf_d=0.03, rho=1.0

------------------ Bounds with CI    ------------------
           CI lower  theta lower     theta  theta upper  CI upper
e401     1306.608818   3593.56223  7710.567004  11935.392833  14114.64269

------------------ Robustness Values ------------------
           H_0    RV (%)  RVa (%)
e401       0.0  6.240295  4.339269"""
    model.sensitivity_summary = summary_string
    
    estimation = {
        'coefficient': 7710.567004,
        'model': model
    }
    
    result = sensitivity_analysis(estimation, cf_y=0.04, cf_d=0.03)
    
    # Should still work with string format
    assert isinstance(result, str)
    assert "e401" in result
    assert "7710.567004" in result


def test_sensitivity_analysis_multiple_rows():
    """Test sensitivity analysis with multiple rows in summary."""
    model = Mock()
    model.sensitivity_analysis = Mock()
    
    # Create summary with multiple rows as string
    summary_string = """================== Sensitivity Analysis ==================

------------------ Scenario          ------------------
Significance Level: level=0.95
Sensitivity parameters: cf_y=0.04; cf_d=0.03, rho=1.0

------------------ Bounds with CI    ------------------
           CI lower  theta lower     theta  theta upper  CI upper
e401     1306.608818   3593.56223  7710.567004  11935.392833  14114.64269
e402     2000.000000   4000.00000  8000.000000  12000.000000  15000.00000

------------------ Robustness Values ------------------
           H_0    RV (%)  RVa (%)
e401       0.0  6.240295  4.339269
e402       0.0  7.500000  5.000000"""
    model.sensitivity_summary = summary_string
    
    estimation = {
        'coefficient': 7710.567004,
        'model': model
    }
    
    result = sensitivity_analysis(estimation, cf_y=0.04, cf_d=0.03)
    
    # Should contain both rows
    assert "e401" in result
    assert "e402" in result
    assert isinstance(result, str)