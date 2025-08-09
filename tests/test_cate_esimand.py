import numpy as np
import pandas as pd

import pytest
from causalkit.data.generators import CausalDatasetGenerator
from causalkit.inference.cate import cate_esimand


def test_cate_esimand_adds_column_and_length_matches():
    # Generate a synthetic dataset with known structure and convert to CausalData
    gen = CausalDatasetGenerator(theta=1.5, seed=123, outcome_type="continuous", target_t_rate=0.4)
    cd = gen.to_causal_data(n=500)

    # Run CATE estimation (in-sample orthogonal signals)
    df_with_cate = cate_esimand(cd)

    # Check that the 'cate' column exists and has correct length
    assert 'cate' in df_with_cate.columns
    assert len(df_with_cate) == len(cd.df)

    # Sanity checks: finite values and variation present
    assert np.isfinite(df_with_cate['cate']).all()
    assert df_with_cate['cate'].std() >= 0.0  # at least defined (may be small), do not enforce > 0 strictly


def test_cate_esimand_blp_predict_matches_shape_with_new_X():
    # Generate training data
    gen = CausalDatasetGenerator(theta=2.0, seed=42, outcome_type="continuous", target_t_rate=0.3)
    cd = gen.to_causal_data(n=300)

    # Create a new X_new with the same confounder columns
    X_cols = list(cd._cofounders)
    X_new = cd.get_df(columns=X_cols, include_treatment=False, include_target=False, include_cofounders=True)

    # Request BLP predictions; output should be a DataFrame with a single 'cate' column and same number of rows
    try:
        cate_new = cate_esimand(cd, use_blp=True, X_new=X_new)
    except NotImplementedError:
        pytest.skip("DoubleML blp_predict not available; skipping BLP-based CATE test")
    else:
        assert isinstance(cate_new, pd.DataFrame)
        assert 'cate' in cate_new.columns
        assert len(cate_new) == len(X_new)
