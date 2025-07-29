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
from causalkit.data import generate_rct_data, CausalData

# Let's generate RCT data
rct_df = generate_rct_data()

# Create a CausalData object
causal_data = CausalData(
    df=rct_df,
    treatment='treatment',
    target='target',
    cofounders=['age', 'cnt_trans', 'platform_Android', 'platform_iOS', 'invited_friend']
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
from causalkit.data import generate_rct_data, CausalData

# Let's generate RCT data
rct_df = generate_rct_data()

# Create a CausalData object
causal_data = CausalData(
    df=rct_df,
    treatment='treatment',
    target='target',
    cofounders=['age', 'cnt_trans', 'platform_Android', 'platform_iOS', 'invited_friend']
)

# Get the full DataFrame
full_df = causal_data.df
# Another way to get the full DataFrame
full_df = causal_data.get_df()
```

### Accessing Treatment, Target, and Cofounders

```python
from causalkit.data import generate_rct_data, CausalData

# Let's generate RCT data
rct_df = generate_rct_data()

# Create a CausalData object
causal_data = CausalData(
    df=rct_df,
    treatment='treatment',
    target='target',
    cofounders=['age', 'cnt_trans', 'platform_Android', 'platform_iOS', 'invited_friend']
)
# Access treatment
causal_data.treatment
# Access target
causal_data.target
# Access cofounders
causal_data.cofounders
```

If you specified multiple columns for any category (e.g., multiple target columns), the corresponding property will return a DataFrame. If you specified a single column, it will return a Series.

### Selective Data Retrieval

The `get_df()` method allows you to retrieve specific columns or column categories:

```python
from causalkit.data import generate_rct_data, CausalData

# Let's generate RCT data
rct_df = generate_rct_data()

# Create a CausalData object
causal_data = CausalData(
    df=rct_df,
    treatment='treatment',
    target='target',
    cofounders=['age', 'cnt_trans', 'platform_Android', 'platform_iOS', 'invited_friend']
)
# Get specific columns
causal_data.get_df(columns=['age'])

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
