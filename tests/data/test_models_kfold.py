import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression

from causalis.eda.eda import PropensityModel, OutcomeModel


def make_synth(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, size=n)
    invited_friend = rng.integers(0, 2, size=n)
    city = rng.choice(['A', 'B', 'C'], size=n, p=[0.5, 0.3, 0.2])
    logits = -2.0 + 0.05 * age + 0.9 * invited_friend + (city == 'B') * 0.3 + (city == 'C') * -0.2
    p = 1 / (1 + np.exp(-logits))
    t = (rng.random(n) < p).astype(int)
    y = 1.0 * t + 0.03 * age + 0.2 * invited_friend + (city == 'B') * 0.5 + rng.normal(0, 1, size=n)
    df = pd.DataFrame({
        'outcome': y,
        'treatment': t,
        'age': age,
        'invited_friend': invited_friend,
        'city': city,
    })
    return df


def test_propensitymodel_from_kfold_logistic():
    df = make_synth(seed=42)
    X = df[['age', 'invited_friend', 'city']]
    t = df['treatment'].values

    pm = PropensityModel.from_kfold(X, t, model=LogisticRegression(max_iter=1000), n_splits=4, random_state=123)

    # Basic validity
    ps = pm.propensity_scores
    assert isinstance(ps, np.ndarray)
    assert ps.shape[0] == len(df)
    assert np.all(ps > 0) and np.all(ps < 1)

    # AUC should be within [0.5, 1]
    auc = pm.roc_auc
    assert 0.5 <= auc <= 1.0

    # SHAP/importance via sklearn coefficients should work
    shap_df = pm.shap
    assert 'feature' in shap_df.columns
    value_columns = [c for c in shap_df.columns if c in ['importance', 'shap_mean']]
    assert len(value_columns) == 1


def test_outcomemodel_from_kfold_linear():
    df = make_synth(seed=7)
    X = df[['age', 'invited_friend', 'city']]
    y = df['outcome'].values

    om = OutcomeModel.from_kfold(X, y, model=LinearRegression(), n_splits=4, random_state=123)

    # Predicted outcomes length matches
    preds = om.predicted_outcomes
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(df)

    # Scores property returns rmse and mae
    scores = om.scores
    assert 'rmse' in scores and 'mae' in scores
    assert np.isfinite(scores['rmse']) and np.isfinite(scores['mae'])
