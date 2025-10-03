import numpy as np
import pandas as pd

from causalkit.eda.eda import CausalEDA, CausalDataLite


def make_dummy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, size=n)
    income = rng.normal(50000, 15000, size=n)
    category = rng.choice(['A', 'B', 'C'], size=n, p=[0.5, 0.3, 0.2])
    # Treatment depends on age and category
    logits = -1.0 + 0.03 * (age - 40) + (category == 'A') * 0.5 + (category == 'C') * -0.3
    ps = 1 / (1 + np.exp(-logits))
    treatment = (rng.uniform(size=n) < ps).astype(int)
    # Outcome depends on age and income, plus noise; exclude treatment to keep it general
    outcome = 0.5 * (age - 40) + 0.00005 * (income - 50000) + rng.normal(0, 1, size=n)

    df = pd.DataFrame({
        't': treatment,
        'y': outcome,
        'age': age,
        'income': income,
        'category': category,
    })
    return df


def test_fit_propensity_and_outcome_without_cat_features():
    df = make_dummy_data()
    data = CausalDataLite(df=df, treatment='t', target='y', confounders=['age', 'income', 'category'])

    eda = CausalEDA(data, n_splits=3, random_state=0)

    # Propensity
    pm = eda.fit_propensity()
    ps = pm.propensity_scores
    assert isinstance(ps, np.ndarray)
    assert ps.shape == (len(df),)
    assert np.isfinite(ps).all()
    assert (ps > 0).all() and (ps < 1).all()

    # Outcome
    om = eda.outcome_fit()
    preds = om.predicted_outcomes
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(df),)
    assert np.isfinite(preds).all()

    # SHAP computations should run without providing cat_features
    shap_ps = pm.shap
    assert not shap_ps.empty
    shap_out = om.shap
    assert not shap_out.empty
