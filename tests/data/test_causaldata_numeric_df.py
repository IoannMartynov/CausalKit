import pandas as pd
import pandas.api.types as pdtypes
import numpy as np
import pytest

from causalis.data.causaldata import CausalData


def test_causaldata_accepts_bool_and_stores_numeric():
    # Create data with boolean outcome, treatment, and one boolean confounder
    n = 20
    rng = np.random.default_rng(123)
    y_bool = rng.integers(0, 2, size=n).astype(bool)
    d_bool = rng.integers(0, 2, size=n).astype(bool)
    x1_bool = rng.integers(0, 2, size=n).astype(bool)
    x2_num = rng.normal(size=n)

    df = pd.DataFrame({
        'y': y_bool,
        'd': d_bool,
        'x1': x1_bool,
        'x2': x2_num,
    })

    cd = CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])

    # All stored columns must be numeric, with no boolean dtype remaining
    for col in cd.df.columns:
        assert pdtypes.is_numeric_dtype(cd.df[col]), f"Column {col} is not numeric: {cd.df[col].dtype}"
        assert not pdtypes.is_bool_dtype(cd.df[col]), f"Column {col} should have been coerced from bool to numeric"

    # Check numeric equivalence of coerced columns (0/1)
    for col in ['y', 'd', 'x1']:
        # Original bools cast to int should equal stored ints
        assert (cd.df[col].to_numpy() == df[col].astype(int).to_numpy()).all()


def test_causaldata_rejects_object_selected_columns():
    # Object in confounder should be rejected
    df = pd.DataFrame({
        'y': [0.1, 0.2, 0.3, 0.4],
        'd': [0, 1, 0, 1],
        'x_bad': ['a', 'b', 'c', 'd'],
    })

    with pytest.raises(ValueError):
        _ = CausalData(df=df, treatment='d', outcome='y', confounders=['x_bad'])
