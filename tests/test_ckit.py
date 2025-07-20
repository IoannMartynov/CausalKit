"""
Test script for the ckit class.
"""

import pandas as pd
import numpy as np
from causalkit.data import generate_rct_data, generate_obs_data, ckit

# Test with RCT data
print("Testing ckit with RCT data:")
rct_df = generate_rct_data(n_users=1000, split=0.5, random_state=42)
print(f"RCT data shape: {rct_df.shape}")
print(f"RCT data columns: {rct_df.columns.tolist()}")

# Create ckit object with RCT data
ck_rct = ckit(
    df=rct_df,
    target='target',
    cofounders=['age', 'invited_friend'],
    treatment='treatment'
)

# Access data through ckit
print("\nAccessing data through ckit:")
print(f"ckit object: {ck_rct}")
print(f"Target column type: {type(ck_rct.target)}")
print(f"Target column shape: {ck_rct.target.shape}")
print(f"Cofounders columns: {ck_rct.cofounders.columns.tolist()}")
print(f"Treatment column mean: {ck_rct.treatment.mean():.2f}")

# Test with observational data
print("\nTesting ckit with observational data:")
obs_df = generate_obs_data(n_users=1000, split=0.1, random_state=42)
print(f"Observational data shape: {obs_df.shape}")
print(f"Observational data columns: {obs_df.columns.tolist()}")

# Create ckit object with observational data
ck_obs = ckit(
    df=obs_df,
    target=None,  # No target column in observational data
    cofounders=['age', 'income', 'education'],
    treatment='treatment'
)

# Access data through ckit
print("\nAccessing data through ckit (observational data):")
print(f"ckit object: {ck_obs}")
print(f"Target column: {ck_obs.target}")
print(f"Cofounders columns: {ck_obs.cofounders.columns.tolist()}")
print(f"Treatment column mean: {ck_obs.treatment.mean():.2f}")

# Test with custom DataFrame
print("\nTesting ckit with custom DataFrame:")
custom_df = pd.DataFrame({
    'user_id': [f'user_{i}' for i in range(5)],
    'ltv': [100, 200, 150, 300, 250],
    'age': [25, 30, 35, 40, 45],
    'invited_friend': [1, 0, 1, 0, 1],
    'treatment': [1, 0, 1, 0, 1]
})
print(f"Custom data:\n{custom_df}")

# Create ckit object with custom data
ck_custom = ckit(
    df=custom_df,
    target='ltv',
    cofounders=['age', 'invited_friend'],
    treatment='treatment'
)

# Access data through ckit
print("\nAccessing data through ckit (custom data):")
print(f"ckit object: {ck_custom}")
print(f"Target column: {ck_custom.target.tolist()}")
print(f"Cofounders columns: {ck_custom.cofounders.columns.tolist()}")
print(f"Treatment column: {ck_custom.treatment.tolist()}")

print("\nTest completed successfully!")