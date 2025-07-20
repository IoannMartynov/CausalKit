"""
Test script to verify that the causalkit package can be installed and imported correctly.
"""

try:
    import causalkit
    print(f"CausalKit version: {causalkit.__version__}")
    print("Successfully imported causalkit")
    
    # Test importing submodules
    from causalkit import data, design, analysis
    print("Successfully imported submodules")
    
    # Test importing specific functions
    from causalkit.data import generate_rct_data
    from causalkit.design import split_traffic
    from causalkit.analysis import ttest
    print("Successfully imported specific functions")
    
    print("All imports successful!")
except ImportError as e:
    print(f"Error importing causalkit: {e}")