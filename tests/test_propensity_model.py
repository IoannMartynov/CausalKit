import numpy as np
import pandas as pd
import pytest

from causalkit.eda.eda import CausalDataLite, CausalEDA
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


def make_dummy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'num1': rng.normal(size=n),
        'num2': rng.normal(size=n),
        'cat1': rng.choice(['A', 'B', 'C'], size=n),
        'cat2': rng.choice(['X', 'Y'], size=n),
    })
    # treatment as a noisy function of features
    logits = 0.5*df['num1'] - 0.3*df['num2'] + (df['cat1'] == 'B').astype(float) - 0.5*(df['cat2'] == 'Y').astype(float)
    p = 1/(1+np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(int)
    y = (df['num1'] + df['num2'] + t + rng.normal(scale=0.5, size=n)).values
    return df, t, y


def test_treatment_features_catboost_alignment():
    df, t, y = make_dummy_data(n=150, seed=1)
    data = CausalDataLite(df=df.assign(treatment=t, outcome=y), treatment='treatment', target='outcome', confounders=['num1', 'num2', 'cat1', 'cat2'])
    eda = CausalEDA(data, ps_model=CatBoostClassifier(thread_count=-1, verbose=False, random_seed=1), n_splits=3, random_state=1)

    _ = eda.fit_propensity()
    feat_df = eda.treatment_features()

    # Columns present
    assert 'feature' in feat_df.columns
    assert 'shap_mean' in feat_df.columns
    assert 'shap_mean_abs' in feat_df.columns

    # Length matches transformed feature space
    assert feat_df.shape[0] == eda._X_for_shap.shape[1]

    # Sorted by magnitude descending
    shap_abs = feat_df['shap_mean_abs'].values
    assert np.all(shap_abs[:-1] >= shap_abs[1:])


def test_treatment_features_sklearn_coef_abs():
    df, t, y = make_dummy_data(n=120, seed=2)
    data = CausalDataLite(df=df.assign(treatment=t, outcome=y), treatment='treatment', target='outcome', confounders=['num1', 'num2', 'cat1', 'cat2'])
    eda = CausalEDA(data, ps_model=LogisticRegression(max_iter=200), n_splits=3, random_state=2)

    _ = eda.fit_propensity()
    feat_df = eda.treatment_features()

    assert 'feature' in feat_df.columns
    assert 'coef_abs' in feat_df.columns
    assert 'shap_mean' not in feat_df.columns  # ensure not mislabeled as SHAP


def test_binary_treatment_validation():
    df, t, y = make_dummy_data(n=100, seed=3)
    # introduce invalid treatment values
    t_bad = t.copy()
    t_bad[::10] = 2
    data = CausalDataLite(df=df.assign(treatment=t_bad, outcome=y), treatment='treatment', target='outcome', confounders=['num1', 'num2', 'cat1', 'cat2'])
    eda = CausalEDA(data, ps_model=CatBoostClassifier(thread_count=-1, verbose=False, random_seed=3), n_splits=3, random_state=3)

    with pytest.raises(ValueError):
        _ = eda.fit_propensity()
