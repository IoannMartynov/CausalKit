# Working with CausalData

The `CausalData` class is a core component of CausalKit that helps you organize and manage your data for causal inference analysis. This guide explains how to use the `CausalData` class effectively.

## Overview

The `CausalData` class wraps a pandas DataFrame and stores metadata about columns for causal inference analysis. It categorizes your data columns into three main types:

- **Target**: The outcome variable(s) you're measuring
- **Treatment**: The intervention or treatment variable(s)
- **Cofounders**: The covariates or confounding variables

This organization makes it easier to perform causal inference analyses and ensures data quality through built-in validation.

## Creating a CausalData Object

You can create a `CausalData` object by passing a pandas DataFrame along with column specifications:

```python
from causalkit.data import CausalData
import pandas as pd

# Create a sample DataFrame
sample_df = pd.DataFrame({
    'user_id': range(100),
    'age': [20 + i % 40 for i in range(100)],
    'treatment': [i % 2 for i in range(100)],
    'conversion': [0.1 + 0.05 * (i % 2) + 0.001 * i for i in range(100)]
})

# Create a CausalData object
sample_causal_data = CausalData(
    df=sample_df,
    target='conversion',
    treatment='treatment',
    cofounders=['age']
)
```

### Requirements and Validation

The `CausalData` class performs several validations when you create an object:

1. The DataFrame cannot contain NaN values
2. All specified columns must exist in the DataFrame
3. Target, treatment, and cofounder columns must contain only numeric values (int or float)

If any of these validations fail, an error will be raised with a descriptive message.

## Accessing Data

Once you've created a `CausalData` object, you can access the data in several ways:

### Accessing the Full DataFrame

```python
from causalkit.data import CausalData
import pandas as pd

# Create a sample DataFrame and CausalData object
sample_df = pd.DataFrame({
    'user_id': range(100),
    'age': [20 + i % 40 for i in range(100)],
    'treatment': [i % 2 for i in range(100)],
    'conversion': [0.1 + 0.05 * (i % 2) + 0.001 * i for i in range(100)]
})

sample_causal_data = CausalData(
    df=sample_df,
    target='conversion',
    treatment='treatment',
    cofounders=['age']
)

# Get the full DataFrame
full_df = sample_causal_data.df
```

### Accessing Specific Column Types

```python
from causalkit.data import CausalData
import pandas as pd

# Create a sample DataFrame and CausalData object
sample_df = pd.DataFrame({
    'user_id': range(100),
    'age': [20 + i % 40 for i in range(100)],
    'treatment': [i % 2 for i in range(100)],
    'conversion': [0.1 + 0.05 * (i % 2) + 0.001 * i for i in range(100)]
})

sample_causal_data = CausalData(
    df=sample_df,
    target='conversion',
    treatment='treatment',
    cofounders=['age']
)

# Get the target variable(s)
target = sample_causal_data.target

# Get the treatment variable(s)
treatment = sample_causal_data.treatment

# Get the cofounders/covariates
cofounders = sample_causal_data.cofounders
```

If you specified multiple columns for any category (e.g., multiple target columns), the corresponding property will return a DataFrame. If you specified a single column, it will return a Series.

### Selective Data Retrieval

The `get_df()` method allows you to retrieve specific columns or column categories:

```python
from causalkit.data import CausalData
import pandas as pd

# Create a sample DataFrame and CausalData object
sample_df = pd.DataFrame({
    'user_id': range(100),
    'age': [20 + i % 40 for i in range(100)],
    'treatment': [i % 2 for i in range(100)],
    'conversion': [0.1 + 0.05 * (i % 2) + 0.001 * i for i in range(100)]
})

sample_causal_data = CausalData(
    df=sample_df,
    target='conversion',
    treatment='treatment',
    cofounders=['age']
)

# Get specific columns by name
specific_cols = sample_causal_data.get_df(columns=['user_id', 'age'])

# Get target and treatment columns
target_treatment = sample_causal_data.get_df(include_target=True, include_treatment=True)

# Get all columns except cofounders
no_cofounders = sample_causal_data.get_df(include_target=True, include_treatment=True, columns=['user_id'])
```

## Working with Generated Data

`CausalData` works seamlessly with CausalKit's data generation functions:

```python
from causalkit.data import generate_rct_data, CausalData

# Generate RCT data
rct_df = generate_rct_data()

# Create a CausalData object
rct_causal_data = CausalData(
    df=rct_df,
    target='target',
    treatment='treatment',
    cofounders=['age', 'invited_friend']
)

# Now you can use this for inference
print(rct_causal_data.target.mean())
print(rct_causal_data.treatment.value_counts())
```

## Multiple Targets and Treatments

`CausalData` supports multiple target and treatment columns:

```python
from causalkit.data import CausalData
import pandas as pd

# Create a sample DataFrame with multiple targets and treatments
multi_df = pd.DataFrame({
    'user_id': range(100),
    'age': [20 + i % 40 for i in range(100)],
    'country': ['US' if i % 3 == 0 else 'UK' if i % 3 == 1 else 'CA' for i in range(100)],
    'previous_purchases': [i % 10 for i in range(100)],
    'email_campaign': [i % 2 for i in range(100)],
    'app_notification': [i % 3 == 0 for i in range(100)],
    'conversion': [0.1 + 0.05 * (i % 2) + 0.001 * i for i in range(100)],
    'revenue': [10 * (i % 5) + 0.5 * i for i in range(100)]
})

# Create a CausalData object with multiple targets and treatments
multi_causal_data = CausalData(
    df=multi_df,
    target=['conversion', 'revenue'],
    treatment=['email_campaign', 'app_notification'],
    cofounders=['age', 'previous_purchases']
)

# Access multiple targets (returns a DataFrame)
targets = multi_causal_data.target

# Access multiple treatments (returns a DataFrame)
treatments = multi_causal_data.treatment
```

## Best Practices

Here are some best practices for working with `CausalData`:

1. **Clean your data before creating a CausalData object**: Handle missing values and ensure numeric columns are properly formatted.

2. **Be explicit about column roles**: Clearly identify which columns are targets, treatments, and cofounders to make your analysis more interpretable.

3. **Use meaningful column names**: This makes your code more readable and helps prevent errors.

4. **Validate your data**: Even though `CausalData` performs basic validation, it's good practice to validate your data before analysis.

## Next Steps

Now that you understand how to use the `CausalData` class, you can:

- Explore the [API Reference](../api/data.md) for detailed documentation
- Check out the [RCT Analysis Example](../examples/rct_analysis.ipynb) for more complex use cases
- Learn about analysis techniques in the [Analysis API](../api/analysis.md)

For any questions or issues, please visit the [GitHub repository](https://github.com/ioannmartynov/causalkit).