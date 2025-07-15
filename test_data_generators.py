import pandas as pd
import numpy as np
import uuid
from causalkit.data import generate_ab_test_data
from causalkit.data.generators import generate_rct_data, generate_obs_data

# Test generate_ab_test_data
print("Testing generate_ab_test_data with default parameters:")
df = generate_ab_test_data(random_state=42)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Group counts: {df['group'].value_counts().to_dict()}")
print(f"Conversion rates: A={df[df['group'] == 'A']['conversion'].mean():.4f}, B={df[df['group'] == 'B']['conversion'].mean():.4f}")
print("\n")

# Test with custom parameters
print("Testing generate_ab_test_data with custom parameters:")
custom_df = generate_ab_test_data(
    n_samples={"A": 5000, "B": 5000},
    conversion_rates={"A": 0.05, "B": 0.07},
    features={
        "age": {"type": "uniform_int", "params": (18, 65)},
        "gender": {"type": "categorical", "params": ["male", "female"]},
        "device": {"type": "categorical", "params": ["desktop", "mobile", "tablet"]},
        "income": {"type": "normal", "params": (60000, 15000)}
    },
    random_state=42
)
print(f"Shape: {custom_df.shape}")
print(f"Columns: {custom_df.columns.tolist()}")
print(f"Group counts: {custom_df['group'].value_counts().to_dict()}")
print(f"Conversion rates: A={custom_df[custom_df['group'] == 'A']['conversion'].mean():.4f}, B={custom_df[custom_df['group'] == 'B']['conversion'].mean():.4f}")
print(f"Device values: {custom_df['device'].unique().tolist()}")

# Test generate_rct_data
print("\nTesting generate_rct_data:")
rct_df = generate_rct_data(n_users=1000, split=0.5, random_state=42)
print(f"Shape: {rct_df.shape}")
print(f"Columns: {rct_df.columns.tolist()}")
print(f"Treatment ratio: {rct_df['treatment'].mean():.2f} (expected: 0.50)")
# Check user_id format (should be UUID)
is_uuid = all(isinstance(uid, str) and len(uid) == 36 for uid in rct_df['user_id'].head())
print(f"User IDs are UUIDs: {is_uuid}")

# Test with custom parameters
rct_custom_df = generate_rct_data(n_users=1000, split=0.3, random_state=123)
treatment_ratio_custom = rct_custom_df['treatment'].mean()
print(f"Custom treatment ratio: {treatment_ratio_custom:.2f} (expected: 0.30)")

# Test generate_obs_data
print("\nTesting generate_obs_data:")
obs_df = generate_obs_data(n_users=1000, split=0.1, random_state=42)
print(f"Shape: {obs_df.shape}")
print(f"Columns: {obs_df.columns.tolist()}")
print(f"Treatment ratio: {obs_df['treatment'].mean():.2f} (expected: 0.10)")
# Check user_id format (should be UUID)
is_uuid = all(isinstance(uid, str) and len(uid) == 36 for uid in obs_df['user_id'].head())
print(f"User IDs are UUIDs: {is_uuid}")

# Test with custom parameters
obs_custom_df = generate_obs_data(n_users=1000, split=0.2, random_state=123)
treatment_ratio_custom = obs_custom_df['treatment'].mean()
print(f"Custom treatment ratio: {treatment_ratio_custom:.2f} (expected: 0.20)")
