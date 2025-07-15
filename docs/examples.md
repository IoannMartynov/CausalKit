# Examples

This page provides examples of using CausalKit for various causal inference tasks.

## A/B Testing Example

This example demonstrates how to generate A/B test data and analyze the results:

```python
import causalkit
from causalkit.data import generate_ab_test_data
from causalkit.analysis import compare_ab

# Generate synthetic A/B test data
df = generate_ab_test_data(
    n_samples={"A": 5000, "B": 5000},
    conversion_rates={"A": 0.10, "B": 0.12},
    random_state=42
)

# Extract control and treatment data
control = df[df['group'] == 'A']['conversion'].values
treatment = df[df['group'] == 'B']['conversion'].values

# Compare the results
compare_ab(control, treatment)
```

## Randomized Controlled Trial (RCT) Example

This example shows how to generate RCT data and analyze it:

```python
import numpy as np
import pandas as pd
from causalkit.data import generate_rct_data
from causalkit.analysis import compare_ab_with_plr

# Generate RCT data
df = generate_rct_data(
    n_users=10000,
    split=0.5,
    target_type="continuous",
    random_state=42
)

# Extract control and treatment data
control = df[df['treatment'] == 0]['outcome'].values
treatment = df[df['treatment'] == 1]['outcome'].values

# Compare using PLR
compare_ab_with_plr(control, treatment)
```

## Traffic Splitting Example

This example demonstrates how to split traffic for an experiment:

```python
import pandas as pd
import numpy as np
from causalkit.design.traffic_splitter import split_traffic

# Create a sample DataFrame
df = pd.DataFrame({
    'user_id': range(1000),
    'age': np.random.normal(30, 5, 1000),
    'gender': np.random.choice(['M', 'F'], 1000),
    'country': np.random.choice(['US', 'UK', 'CA', 'AU'], 1000)
})

# Split into control and treatment groups with stratification
control_df, treatment_df = split_traffic(
    df,
    split_ratio=0.5,
    stratify_column='country',
    random_state=42
)

# Verify the distribution of the stratification variable
print("Control group country distribution:")
print(control_df['country'].value_counts(normalize=True))

print("\nTreatment group country distribution:")
print(treatment_df['country'].value_counts(normalize=True))
```

## More Examples

For more examples, check out the Jupyter notebooks in the [examples directory](https://github.com/yourusername/causalkit/tree/main/causalkit/examples) of the repository:

- [A/B Testing with T-Test](https://github.com/yourusername/causalkit/blob/main/causalkit/examples/ttest.ipynb)
- [Traffic Splitting](https://github.com/yourusername/causalkit/blob/main/causalkit/examples/traffic_splitting_example.ipynb)
- [Double Machine Learning with PLR](https://github.com/yourusername/causalkit/blob/main/causalkit/examples/dml_pls_ttest.ipynb)