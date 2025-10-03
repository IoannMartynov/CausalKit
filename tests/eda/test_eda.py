import numpy as np
import pandas as pd

from causalkit.data.causaldata import CausalData
from causalkit.eda import CausalEDA


def make_synth(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, size=n)
    invited_friend = rng.integers(0, 2, size=n)
    # treatment depends on covariates
    logits = -2.0 + 0.04 * age + 0.8 * invited_friend
    p = 1 / (1 + np.exp(-logits))
    t = (rng.random(n) < p).astype(int)
    # outcome depends on t and covariates
    y = 0.5 * t + 0.02 * age + 0.3 * invited_friend + rng.normal(0, 1, size=n)
    df = pd.DataFrame({
        'outcome': y,
        'treatment': t,
        'age': age,
        'invited_friend': invited_friend.astype(float),  # keep numeric to satisfy CausalData
    })
    return df




def test_fit_propensity_and_confounders_means_direct_calls():
    df = make_synth(seed=42)
    cd = CausalData(df=df, treatment='treatment', outcome='outcome', confounders=['age', 'invited_friend'])

    eda = CausalEDA(cd)
    ps_model = eda.fit_propensity()
    
    # Test that fit_propensity returns a PropensityModel
    from causalkit.eda.eda import PropensityModel
    assert isinstance(ps_model, PropensityModel)
    
    # Test propensity scores are accessible and valid
    ps = ps_model.propensity_scores
    assert ps.shape[0] == df.shape[0]
    assert np.all(ps > 0) and np.all(ps < 1)
    
    # Test PropensityModel methods work
    auc = ps_model.roc_auc
    assert 0.5 <= auc <= 1.0
    
    # Test positivity check
    pos_check = ps_model.positivity_check()
    assert 'bounds' in pos_check and 'share_below' in pos_check and 'share_above' in pos_check
    
    # Test SHAP values property
    shap_df = ps_model.shap
    assert isinstance(shap_df, pd.DataFrame)
    assert 'feature' in shap_df.columns
    
    # Test old interface still works (backward compatibility)
    auc_old = eda.confounders_roc_auc(ps)
    assert 0.5 <= auc_old <= 1.0
    assert abs(auc - auc_old) < 1e-10  # Should be the same

    bal = eda.confounders_means()
    assert not bal.empty
    # verify comprehensive balance columns with new names
    expected_columns = {'mean_t_0', 'mean_t_1', 'abs_diff', 'smd'}
    assert expected_columns.issubset(set(bal.columns))
    # verify that index contains confounders names
    assert bal.index.name == 'confounders'


def test_treatment_features():
    """Test the treatment_features() method with CatBoost (default) model."""
    df = make_synth(seed=42)
    cd = CausalData(df=df, treatment='treatment', outcome='outcome', confounders=['age', 'invited_friend'])

    eda = CausalEDA(cd)
    
    # Test error case: calling treatment_features() before fit_propensity()
    try:
        eda.treatment_features()
        assert False, "Should raise RuntimeError when no model is fitted"
    except RuntimeError as e:
        assert "No fitted propensity model found" in str(e)
    
    # Fit propensity model first
    ps_model = eda.fit_propensity()
    
    # Now test successful feature importance/SHAP extraction
    features_df = eda.treatment_features()
    
    # Check return type and structure
    assert isinstance(features_df, pd.DataFrame)
    assert 'feature' in features_df.columns
    
    # Check for either 'importance' (sklearn models) or 'shap_mean' (CatBoost models)
    value_columns = [col for col in features_df.columns if col in ['importance', 'shap_mean']]
    assert len(value_columns) == 1, f"Expected exactly one value column, got {value_columns}"
    value_column = value_columns[0]
    
    # Check that we have the expected features
    expected_features = {'age', 'invited_friend'}
    actual_features = set(features_df['feature'].tolist())
    assert expected_features.issubset(actual_features), f"Expected {expected_features}, got {actual_features}"
    
    # Check that values are numeric
    assert features_df[value_column].dtype in [np.float64, np.float32, float]
    
    # Check validation based on column type
    if value_column == 'importance':
        # Regular importance values should be non-negative
        assert (features_df[value_column] >= 0).all()
        # Check that results are sorted by importance (descending)
        values = features_df[value_column].tolist()
        assert values == sorted(values, reverse=True)
    elif value_column == 'shap_mean':
        # SHAP values can be negative - check that results are sorted by absolute value (descending)
        abs_values = features_df[value_column].abs().tolist()
        assert abs_values == sorted(abs_values, reverse=True)
    
    # Check that we have some reasonable number of features (at least our 2 input features)
    assert len(features_df) >= 2
