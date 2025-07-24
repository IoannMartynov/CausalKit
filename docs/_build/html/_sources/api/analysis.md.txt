# Analysis Module

The `causalkit.analysis` module provides statistical analysis tools for causal inference.

## Overview

This module includes functions for:

- Performing t-tests on causaldata objects to compare target variables between treatment groups
- Calculating p-values, absolute differences, and relative differences with confidence intervals

## T-Test Analysis

The `ttest` function performs a t-test on a causaldata object to compare the target variable between treatment groups. This is particularly useful for analyzing the results of A/B tests or randomized controlled trials (RCTs).

### Key Features

- Compares means between treatment and control groups
- Calculates p-values to determine statistical significance
- Provides absolute difference between group means with confidence intervals
- Calculates relative difference (percentage change) with confidence intervals
- Supports customizable confidence levels

### When to Use T-Tests

T-tests are appropriate when:

- You have a binary treatment variable (e.g., control vs. treatment)
- Your target variable is continuous or binary
- You want to determine if there's a statistically significant difference between groups
- You need to quantify the magnitude of the effect with confidence intervals

### Example Usage

```python
from causalkit.data import generate_rct_data, CausalData
from causalkit.inference import ttest

# Generate sample RCT data
df = generate_rct_data(
    n_users=10000,
    split=0.5,
    target_type="normal",
    target_params={"mean": {"A": 10.0, "B": 10.5}, "std": 2.0},
    random_state=42
)

# Create causaldata object
ck = CausalData(
    df=df,
    target='target',
    treatment='treatment'
)

# Perform t-test with 95% confidence level
results = ttest(ck, confidence_level=0.95)

# Print results
print(f"P-value: {results['p_value']:.4f}")
print(f"Absolute difference: {results['absolute_difference']:.4f}")
print(f"Absolute CI: {results['absolute_ci']}")
print(f"Relative difference: {results['relative_difference']:.2f}%")
print(f"Relative CI: {results['relative_ci']}")
```

### Interpreting Results

- **p-value**: Indicates the probability of observing the data if there is no true difference between groups. A small p-value (typically < 0.05) suggests that the observed difference is statistically significant.
- **absolute_difference**: The raw difference between the treatment and control means.
- **absolute_ci**: Confidence interval for the absolute difference. If this interval does not include zero, the difference is statistically significant.
- **relative_difference**: The percentage change relative to the control group mean.
- **relative_ci**: Confidence interval for the relative difference.

## API Reference

```{eval-rst}
.. automodule:: causalkit.inference.ttest
   :members: ttest
   :undoc-members:
   :show-inheritance:
```