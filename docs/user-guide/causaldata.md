# CausalData at a glance

CausalData is the light-weight input container used across CausalKit. It wraps a pandas DataFrame and records which columns are the outcome (target), the treatment, and the confounders.

## Quick start

```python
from causalkit.data import generate_rct_data, CausalData

# Example data
df = generate_rct_data(n_users=5_000)

# Declare column roles
causal_data = CausalData(
    df=df,
    treatment='treatment',
    outcome='outcome',
    confounders=['age', 'cnt_trans', 'platform_Android', 'platform_iOS', 'invited_friend']
)
```

Note: Internally, the stored DataFrame is trimmed to only these columns: [outcome, treatment, confounders].

## API essentials

- Init parameters
  - df: pandas DataFrame (no NaNs)
  - treatment: name of the treatment column (numeric)
  - outcome: name of the outcome/target column (numeric)
  - confounders: one or more confounder column names (numeric)

- Properties
  - target: pandas Series (the outcome)
  - outcome: alias of target
  - treatment: pandas Series
  - confounders: list[str] of confounder column names

- Method
  - get_df(columns=None, include_treatment=True, include_target=True, include_confounders=True) -> DataFrame
    Selects columns by name and/or by role. Returns a copy.

## Validation (on construction)

1) No missing values anywhere in df.
2) All referenced columns must exist.
3) Outcome, treatment, and confounders must be numeric (int/float).
4) None of these columns can be constant (zero variance).
5) Any two used columns having identical values is disallowed (raises ValueError).
6) Duplicate rows across the used columns trigger a warning (not an error).

## Common snippets

```python
from causalkit.data import generate_rct_data, CausalData

df = generate_rct_data(n_users=1_000)
causal_data = CausalData(
    df=df,
    treatment='treatment',
    outcome='outcome',
    confounders=['age', 'cnt_trans', 'platform_Android', 'platform_iOS', 'invited_friend']
)

# Access pieces
causal_data.treatment     # Series
causal_data.target        # Series
causal_data.confounders   # list[str]

# Full data used by CausalData
default_df = causal_data.df           # or equivalently
default_df = causal_data.get_df()

# DataFrame of only confounders
X = causal_data.get_df(include_target=False, include_treatment=False)
# or
X = causal_data.df[causal_data.confounders]

# Select a subset by name(s)
small = causal_data.get_df(columns=['age'])
```

## Tips

- For categorical confounders, encode them numerically (e.g., one-hot) before creating CausalData.
- If you see the duplicate-rows warning, consider deduplicating if duplicates are unintended.
- __repr__ shows the stored shape and declared roles for quick inspection.

