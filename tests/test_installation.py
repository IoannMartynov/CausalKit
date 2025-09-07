"""
Tests to verify that the causalkit package can be installed and imported correctly.
"""

import pytest


def test_import_causalkit():
    """Test that the causalkit package can be imported."""
    try:
        import causalkit
        # Check that __version__ attribute exists and is a string
        assert hasattr(causalkit, '__version__')
        assert isinstance(causalkit.__version__, str)
    except ImportError as e:
        pytest.fail(f"Failed to import causalkit: {e}")


def test_import_submodules():
    """Test that causalkit submodules can be imported."""
    try:
        from causalkit import data, design, inference
        # Check that the imported objects are modules
        assert data.__name__ == 'causalkit.data'
        assert design.__name__ == 'causalkit.design'
        assert inference.__name__ == 'causalkit.inference'
    except ImportError as e:
        pytest.fail(f"Failed to import causalkit submodules: {e}")


def test_import_specific_functions():
    """Test that specific functions from causalkit can be imported."""
    try:
        from causalkit.data import generate_rct
        from causalkit.design import split_traffic
        from causalkit.inference import ttest
        
        # Check that the imported objects are callable
        assert callable(generate_rct)
        assert callable(split_traffic)
        assert callable(ttest)
    except ImportError as e:
        pytest.fail(f"Failed to import specific functions from causalkit: {e}")


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main(["-xvs", __file__])