# Getting Started with CausalKit

This guide will help you get started with CausalKit by walking through some basic examples.

## Basic Workflow

A typical workflow with CausalKit involves:

1. Generating or loading data
2. Designing and implementing an experiment
3. Analyzing the results

Let's walk through each step with examples.

## Data Generation

CausalKit provides several functions for generating synthetic data for causal inference tasks.

### Generating A/B Test Data

```python
from causalkit.data import generate_ab_test_data

# Generate A/B test data with default parameters
df = generate_ab_test_data()

# Generate A/B test data with custom parameters
df_custom = generate_ab_test_data(
    n_samples={"A": 5000, "B": 5000},
    conversion_rates={"A": 0.10, "B": 0.12},
    random_state=42
)

print(df_custom.head())
```

### Generating Randomized Controlled Trial (RCT) Data

```python
from causalkit.data import generate_rct_data

# Generate RCT data with default parameters
df = generate_rct_data()

# Generate RCT data with custom parameters
df_custom = generate_rct_data(
    n_users=10000,
    split=0.5,
    target_type="binary",
    random_state=42
)

print(df_custom.head())
```

## Experimental Design

### Splitting Traffic

CausalKit provides utilities for splitting traffic data for experiments.

```python
import pandas as pd
from causalkit.design.traffic_splitter import split_traffic

# Create a sample DataFrame
df = pd.DataFrame({
    'user_id': range(1000),
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.choice(['A', 'B', 'C'], 1000)
})

# Split into training and test sets (70% / 30%)
train_df, test_df = split_traffic(df, split_ratio=0.7, random_state=42)

# Split into training, validation, and test sets (60% / 20% / 20%)
train_df, val_df, test_df = split_traffic(df, split_ratio=[0.6, 0.2], random_state=42)

# Stratified split based on a categorical feature
train_df, test_df = split_traffic(df, split_ratio=0.7, stratify_column='feature_2', random_state=42)
```

## Analysis

### Comparing A/B Test Results

```python
import numpy as np
from causalkit.inference import compare_ab

# Generate some sample data
control = np.random.normal(10, 2, 1000)  # Control group data
treatment = np.random.normal(10.5, 2, 1000)  # Treatment group data

# Compare the results
compare_ab(control, treatment)
```

### Advanced Analysis with PLR

```python
from causalkit.inference import compare_ab_with_plr

# Compare using Partial Linear Regression
compare_ab_with_plr(control, treatment)
```

## Next Steps

Now that you're familiar with the basic functionality of CausalKit, you can:

- Explore the [API Reference](../api/data.md) for detailed documentation of all functions
- Check out the [Examples](../examples.md) for more complex use cases
- Read about advanced topics in causal inference in the user guide

For any questions or issues, please visit the [GitHub repository](https://github.com/ioannmartynov/causalkit).