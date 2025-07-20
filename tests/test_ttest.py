"""
Test script to verify the ttest function in the analysis module.
"""

import pandas as pd
import numpy as np
from causalkit.data import causaldata
from causalkit.analysis.ttest import ttest

# Set random seed for reproducibility
np.random.seed(42)

# Create a test DataFrame with a known effect size
n = 1000
control_mean = 10.0
treatment_effect = 2.0
treatment_mean = control_mean + treatment_effect

# Create data with treatment effect
df = pd.DataFrame({
    'user_id': range(1, n + 1),
    'treatment': np.random.choice([0, 1], size=n),
    'age': np.random.randint(18, 65, size=n),
    'gender': np.random.choice(['M', 'F'], size=n),
})

# Generate target variable with treatment effect
df['target'] = np.where(
    df['treatment'] == 1,
    np.random.normal(treatment_mean, 2.0, size=n),  # Treatment group
    np.random.normal(control_mean, 2.0, size=n)     # Control group
)

# Create a causaldata object
ck = causaldata(
    df=df,
    target='target',
    cofounders=['age', 'gender'],
    treatment='treatment'
)

# Test 1: Basic functionality
print("Test 1: Basic functionality")
result = ttest(ck)
print(f"p_value: {result['p_value']}")
print(f"absolute_difference: {result['absolute_difference']}")
print(f"absolute_ci: {result['absolute_ci']}")
print(f"relative_difference: {result['relative_difference']}")
print(f"relative_ci: {result['relative_ci']}")
print()

# Test 2: Check if absolute difference is close to the expected treatment effect
print("Test 2: Check if absolute difference is close to the expected treatment effect")
expected_diff = treatment_effect
actual_diff = result['absolute_difference']
diff_error = abs(actual_diff - expected_diff)
print(f"Expected difference: {expected_diff}")
print(f"Actual difference: {actual_diff}")
print(f"Error: {diff_error}")
print(f"Test passed: {diff_error < 0.5}")  # Allow for some random variation
print()

# Test 3: Check if confidence intervals contain the true effect
print("Test 3: Check if confidence intervals contain the true effect")
lower_bound, upper_bound = result['absolute_ci']
contains_true_effect = lower_bound <= expected_diff <= upper_bound
print(f"Confidence interval: ({lower_bound:.4f}, {upper_bound:.4f})")
print(f"True effect: {expected_diff}")
print(f"CI contains true effect: {contains_true_effect}")
print()

# Test 4: Check relative difference
print("Test 4: Check relative difference")
expected_rel_diff = (treatment_effect / control_mean) * 100
actual_rel_diff = result['relative_difference']
rel_diff_error = abs(actual_rel_diff - expected_rel_diff)
print(f"Expected relative difference: {expected_rel_diff:.2f}%")
print(f"Actual relative difference: {actual_rel_diff:.2f}%")
print(f"Error: {rel_diff_error:.2f}%")
print(f"Test passed: {rel_diff_error < 5}")  # Allow for some random variation
print()

# Test 5: Different confidence level
print("Test 5: Different confidence level")
result_90 = ttest(ck, confidence_level=0.90)
result_99 = ttest(ck, confidence_level=0.99)

# 90% CI should be narrower than 95% CI (default)
ci_width_90 = result_90['absolute_ci'][1] - result_90['absolute_ci'][0]
ci_width_95 = result['absolute_ci'][1] - result['absolute_ci'][0]
ci_width_99 = result_99['absolute_ci'][1] - result_99['absolute_ci'][0]

print(f"90% CI width: {ci_width_90:.4f}")
print(f"95% CI width: {ci_width_95:.4f}")
print(f"99% CI width: {ci_width_99:.4f}")
print(f"Test passed: {ci_width_90 < ci_width_95 < ci_width_99}")
print()

# Test 6: Error handling - no treatment
print("Test 6: Error handling - no treatment")
ck_no_treatment = causaldata(
    df=df,
    target='target',
    cofounders=['age', 'gender']
)
try:
    result = ttest(ck_no_treatment)
    print("Error: Test failed - should have raised ValueError")
except ValueError as e:
    print(f"Success: Correctly raised ValueError: {e}")
print()

# Test 7: Error handling - no target
print("Test 7: Error handling - no target")
ck_no_target = causaldata(
    df=df,
    cofounders=['age', 'gender'],
    treatment='treatment'
)
try:
    result = ttest(ck_no_target)
    print("Error: Test failed - should have raised ValueError")
except ValueError as e:
    print(f"Success: Correctly raised ValueError: {e}")
print()

# Test 8: Error handling - non-binary treatment
print("Test 8: Error handling - non-binary treatment")
df_multi = df.copy()
df_multi['treatment'] = np.random.choice([0, 1, 2], size=n)
ck_multi = causaldata(
    df=df_multi,
    target='target',
    cofounders=['age', 'gender'],
    treatment='treatment'
)
try:
    result = ttest(ck_multi)
    print("Error: Test failed - should have raised ValueError")
except ValueError as e:
    print(f"Success: Correctly raised ValueError: {e}")
print()

print("All tests completed.")