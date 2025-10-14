import numpy as np
import pandas as pd
import pytest

from causalis.data.generators import CausalDatasetGenerator
from causalis.inference.gate import gate_esimand


def test_gate_esimand_default_quintiles_shape_and_columns():
    gen = CausalDatasetGenerator(theta=1.0, seed=123, outcome_type="continuous", target_d_rate=0.4)
    cd = gen.to_causal_data(n=600)

    res = gate_esimand(cd, n_groups=5)

    # columns
    expected_cols = {"group", "n", "theta", "std_error", "p_value", "ci_lower", "ci_upper"}
    assert expected_cols.issubset(set(res.columns))

    # groups count should be <= n_groups (duplicates="drop" can reduce)
    assert 1 <= res.shape[0] <= 5

    # sanity checks
    assert (res["n"] > 0).all()
    assert np.isfinite(res[["theta", "p_value", "ci_lower", "ci_upper"]]).to_numpy().all()


def test_gate_esimand_with_custom_groups_matches_group_counts():
    gen = CausalDatasetGenerator(theta=1.5, seed=7, outcome_type="continuous", target_d_rate=0.35)
    cd = gen.to_causal_data(n=500)

    # Create a simple custom grouping by a confounder (e.g., x1 median split)
    X_cols = list(cd.confounders)
    X = cd.get_df(columns=X_cols, include_treatment=False, include_target=False, include_confounders=True)
    median = X.iloc[:, 0].median()
    g = (X.iloc[:, 0] > median).astype(int)

    res = gate_esimand(cd, groups=pd.Series(g))

    # Expect two groups 0 and 1
    assert set(res["group"].astype(int).unique()) <= {0, 1}
    assert res.shape[0] <= 2
    assert res["n"].sum() == len(cd.df)
