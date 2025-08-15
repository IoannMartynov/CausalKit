#!/usr/bin/env python3
"""
Verification script to confirm that missingness_report has been removed from EDA.
This script demonstrates that:
1. CausalData cannot contain NaN values (as expected)
2. CausalEDA no longer has missingness_report method
3. EDA workflow works correctly without missingness functionality
"""

import numpy as np
import pandas as pd
from causalkit.data import CausalData
from causalkit.eda import CausalEDA

def test_causal_data_no_nan():
    """Test that CausalData rejects NaN values"""
    print("1. Testing CausalData NaN rejection...")
    
    # Create data with NaN values
    df_with_nan = pd.DataFrame({
        'y': [1.0, 2.0, np.nan, 4.0],
        't': [0, 1, 0, 1],
        'x': [1.5, 2.5, 3.5, 4.5]
    })
    
    try:
        causal_data = CausalData(df=df_with_nan, treatment='t', outcome='y', cofounders=['x'])
        print("   ERROR: CausalData should have rejected NaN values!")
        return False
    except ValueError as e:
        if "NaN values" in str(e):
            print("   ✓ CausalData correctly rejects NaN values")
            return True
        else:
            print(f"   ERROR: Unexpected error: {e}")
            return False

def test_missingness_method_removed():
    """Test that missingness_report method is no longer available"""
    print("\n2. Testing missingness_report method removal...")
    
    # Create valid data without NaN
    df = pd.DataFrame({
        'y': [1.0, 2.0, 3.0, 4.0],
        't': [0, 1, 0, 1],
        'x': [1.5, 2.5, 3.5, 4.5]
    })
    
    causal_data = CausalData(df=df, treatment='t', outcome='y', cofounders=['x'])
    eda = CausalEDA(causal_data)
    
    # Check if missingness_report method exists
    if hasattr(eda, 'missingness_report'):
        print("   ERROR: missingness_report method still exists!")
        return False
    else:
        print("   ✓ missingness_report method has been removed")
        return True

def test_eda_workflow():
    """Test that EDA workflow works without missingness functionality"""
    print("\n3. Testing EDA workflow without missingness functionality...")
    
    # Create larger, more realistic dataset
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'y': np.random.normal(10, 3, n),
        't': np.random.binomial(1, 0.4, n),
        'age': np.random.normal(35, 10, n),
        'income': np.random.normal(50000, 15000, n)
    })
    
    causal_data = CausalData(df=df, treatment='t', outcome='y', cofounders=['age', 'income'])
    eda = CausalEDA(causal_data)
    
    try:
        # Test data health check (should still work)
        health = eda.data_health_check()
        print(f"   ✓ Data health check works: {len(health)} metrics")
        
        # Test design report (should work without 'missing' key)
        report = eda.design_report()
        expected_keys = {'health', 'summaries', 'treat_auc', 'positivity', 'balance', 'weights'}
        actual_keys = set(report.keys())
        
        if expected_keys == actual_keys:
            print("   ✓ Design report works correctly without 'missing' key")
            print(f"   ✓ Report contains keys: {sorted(actual_keys)}")
            return True
        else:
            print(f"   ERROR: Unexpected keys in design report")
            print(f"   Expected: {sorted(expected_keys)}")
            print(f"   Actual: {sorted(actual_keys)}")
            return False
            
    except Exception as e:
        print(f"   ERROR: EDA workflow failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("Verifying removal of missingness_report from EDA...")
    print("=" * 60)
    
    tests = [
        test_causal_data_no_nan,
        test_missingness_method_removed,
        test_eda_workflow
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED - missingness functionality successfully removed!")
        print("\nSummary of changes:")
        print("- CausalData continues to reject NaN values (as designed)")
        print("- missingness_report method removed from CausalEDA")
        print("- EDA workflow works correctly without missingness functionality")
        print("- design_report no longer includes 'missing' key")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)