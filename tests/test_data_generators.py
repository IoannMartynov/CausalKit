"""
Test script to verify the data generator functions in the causalkit.data.generators module.
"""

import pandas as pd
import numpy as np
from causalkit.data.generators import generate_rct_data, generate_obs_data

# Set random seed for reproducibility
np.random.seed(42)

# Test 1: generate_rct_data with default parameters
print("Test 1: generate_rct_data with default parameters")
rct_df = generate_rct_data(random_state=42)
print(f"Shape: {rct_df.shape}")
print(f"Columns: {rct_df.columns.tolist()}")
print(f"Treatment ratio: {rct_df['treatment'].mean():.2f} (expected: 0.50)")
print(f"Target mean (control): {rct_df[rct_df['treatment'] == 0]['target'].mean():.4f}")
print(f"Target mean (treatment): {rct_df[rct_df['treatment'] == 1]['target'].mean():.4f}")
# Check user_id format (should be UUID)
is_uuid = all(isinstance(uid, str) and len(uid) == 36 for uid in rct_df['user_id'].head())
print(f"User IDs are UUIDs: {is_uuid}")
print()

# Test 2: generate_rct_data with custom parameters (binary target)
print("Test 2: generate_rct_data with custom parameters (binary target)")
rct_binary_df = generate_rct_data(
    n_users=1000,
    split=0.3,
    target_type="binary",
    target_params={"p": {"A": 0.15, "B": 0.20}},
    random_state=42
)
treatment_ratio = rct_binary_df['treatment'].mean()
control_conversion = rct_binary_df[rct_binary_df['treatment'] == 0]['target'].mean()
treatment_conversion = rct_binary_df[rct_binary_df['treatment'] == 1]['target'].mean()
print(f"Shape: {rct_binary_df.shape}")
print(f"Treatment ratio: {treatment_ratio:.2f} (expected: 0.30)")
print(f"Control conversion rate: {control_conversion:.4f} (expected: ~0.15)")
print(f"Treatment conversion rate: {treatment_conversion:.4f} (expected: ~0.20)")
print(f"Test passed: {abs(treatment_ratio - 0.3) < 0.01 and abs(control_conversion - 0.15) < 0.02 and abs(treatment_conversion - 0.20) < 0.02}")
print()

# Test 3: generate_rct_data with normal target
print("Test 3: generate_rct_data with normal target")
rct_normal_df = generate_rct_data(
    n_users=1000,
    split=0.5,
    target_type="normal",
    target_params={"mean": {"A": 10.0, "B": 12.0}, "std": 2.0},
    random_state=42
)
control_mean = rct_normal_df[rct_normal_df['treatment'] == 0]['target'].mean()
treatment_mean = rct_normal_df[rct_normal_df['treatment'] == 1]['target'].mean()
control_std = rct_normal_df[rct_normal_df['treatment'] == 0]['target'].std()
treatment_std = rct_normal_df[rct_normal_df['treatment'] == 1]['target'].std()
print(f"Control mean: {control_mean:.4f} (expected: ~10.0)")
print(f"Treatment mean: {treatment_mean:.4f} (expected: ~12.0)")
print(f"Control std: {control_std:.4f} (expected: ~2.0)")
print(f"Treatment std: {treatment_std:.4f} (expected: ~2.0)")
print(f"Test passed: {abs(control_mean - 10.0) < 0.2 and abs(treatment_mean - 12.0) < 0.2 and abs(control_std - 2.0) < 0.2 and abs(treatment_std - 2.0) < 0.2}")
print()

# Test 4: generate_rct_data with non-normal target
print("Test 4: generate_rct_data with non-normal target")
rct_nonnormal_df = generate_rct_data(
    n_users=1000,
    split=0.5,
    target_type="nonnormal",
    target_params={"shape": 3.0, "scale": {"A": 1.0, "B": 1.5}},
    random_state=42
)
control_mean = rct_nonnormal_df[rct_nonnormal_df['treatment'] == 0]['target'].mean()
treatment_mean = rct_nonnormal_df[rct_nonnormal_df['treatment'] == 1]['target'].mean()
# Expected means for gamma distribution: shape * scale
expected_control_mean = 3.0 * 1.0
expected_treatment_mean = 3.0 * 1.5
print(f"Control mean: {control_mean:.4f} (expected: ~{expected_control_mean})")
print(f"Treatment mean: {treatment_mean:.4f} (expected: ~{expected_treatment_mean})")
print(f"Test passed: {abs(control_mean - expected_control_mean) < 0.3 and abs(treatment_mean - expected_treatment_mean) < 0.3}")
print()

# Test 5: generate_rct_data with invalid target_type
print("Test 5: generate_rct_data with invalid target_type")
try:
    invalid_df = generate_rct_data(target_type="invalid", random_state=42)
    print("Error: Test failed - should have raised ValueError")
except ValueError as e:
    print(f"Success: Correctly raised ValueError: {e}")
print()

# Test 6: generate_obs_data with default parameters
print("Test 6: generate_obs_data with default parameters")
obs_df = generate_obs_data(random_state=42)
print(f"Shape: {obs_df.shape}")
print(f"Columns: {obs_df.columns.tolist()}")
print(f"Treatment ratio: {obs_df['treatment'].mean():.2f} (expected: 0.10)")
# Check user_id format (should be UUID)
is_uuid = all(isinstance(uid, str) and len(uid) == 36 for uid in obs_df['user_id'].head())
print(f"User IDs are UUIDs: {is_uuid}")
print()

# Test 7: generate_obs_data with custom parameters
print("Test 7: generate_obs_data with custom parameters")
obs_custom_df = generate_obs_data(n_users=1000, split=0.2, random_state=123)
treatment_ratio = obs_custom_df['treatment'].mean()
print(f"Shape: {obs_custom_df.shape}")
print(f"Treatment ratio: {treatment_ratio:.2f} (expected: 0.20)")
print(f"Test passed: {abs(treatment_ratio - 0.2) < 0.01}")
print()

# Test 8: Check covariate distributions in generate_obs_data
print("Test 8: Check covariate distributions in generate_obs_data")
obs_check_df = generate_obs_data(n_users=5000, split=0.1, random_state=42)
# Check if education levels are distributed as expected
education_counts = obs_check_df['education'].value_counts(normalize=True)
print(f"Education distribution: {education_counts.to_dict()}")
print(f"Expected distribution: high_school=0.3, bachelor=0.4, master=0.2, phd=0.1")
# Check if regions are distributed evenly (5 regions)
region_counts = obs_check_df['region'].value_counts(normalize=True)
print(f"Region distribution: {region_counts.to_dict()}")
print(f"Expected distribution: approximately 0.2 for each region")
print()

print("All tests completed.")
