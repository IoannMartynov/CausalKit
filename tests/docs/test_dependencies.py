"""
Test that required dependencies can be imported.
"""

def test_sklearn_import():
    """Test that sklearn can be imported."""
    try:
        import sklearn
        assert sklearn.__version__
        print(f"Successfully imported sklearn version {sklearn.__version__}")
    except ImportError:
        assert False, "Failed to import sklearn"

def test_doubleml_import():
    """Test that doubleml can be imported."""
    try:
        import doubleml
        assert doubleml.__version__
        print(f"Successfully imported doubleml version {doubleml.__version__}")
    except ImportError:
        assert False, "Failed to import doubleml"

if __name__ == "__main__":
    test_sklearn_import()
    test_doubleml_import()
    print("All dependency tests passed!")