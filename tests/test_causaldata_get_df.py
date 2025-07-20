"""
Test script to verify the get_df method in causaldata class.
"""

import pandas as pd
import numpy as np
from causalkit.data import causaldata

# Create a test DataFrame
df = pd.DataFrame({
    'user_id': range(1, 101),
    'treatment': np.random.choice([0, 1], size=100),
    'age': np.random.randint(18, 65, size=100),
    'gender': np.random.choice(['M', 'F'], size=100),
    'invited_friend': np.random.choice([0, 1], size=100),
    'target': np.random.normal(0, 1, size=100)
})

# Create a causaldata object
ck = causaldata(
    df=df,
    target='target',
    cofounders=['age', 'invited_friend'],
    treatment='treatment'
)

# Test 1: Get the entire DataFrame
print("Test 1: Get the entire DataFrame")
result_df = ck.get_df()
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print(f"Is a copy: {result_df is not df}")
print()

# Test 2: Get specific columns
print("Test 2: Get specific columns")
result_df = ck.get_df(columns=['user_id', 'gender'])
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print()

# Test 3: Get target columns
print("Test 3: Get target columns")
result_df = ck.get_df(include_target=True)
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print()

# Test 4: Get cofounder columns
print("Test 4: Get cofounder columns")
result_df = ck.get_df(include_cofounders=True)
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print()

# Test 5: Get treatment columns
print("Test 5: Get treatment columns")
result_df = ck.get_df(include_treatment=True)
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print()

# Test 6: Get combination of columns
print("Test 6: Get combination of columns")
result_df = ck.get_df(
    columns=['user_id', 'gender'],
    include_target=True,
    include_treatment=True
)
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print()

# Test 7: Handle duplicate columns
print("Test 7: Handle duplicate columns")
result_df = ck.get_df(
    columns=['age', 'gender'],
    include_cofounders=True  # 'age' is also in cofounders
)
print(f"Shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")
print()

# Test 8: Error handling for non-existent columns
print("Test 8: Error handling for non-existent columns")
try:
    result_df = ck.get_df(columns=['non_existent_column'])
    print("Error: Test failed - should have raised ValueError")
except ValueError as e:
    print(f"Success: Correctly raised ValueError: {e}")
print()

print("All tests completed.")