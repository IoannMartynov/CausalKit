import numpy as np
import pandas as pd

from causalkit.eda.eda import CausalDataLite, CausalEDA
from catboost import CatBoostClassifier


def make_dummy_data(n=160, seed=123):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(loc=0.5, scale=1.2, size=n),
        'c1': rng.choice(['A', 'B'], size=n),
    })
    logits = 0.7 * df['x1'] - 0.4 * df['x2'] + 0.8 * (df['c1'] == 'B').astype(float)
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(int)
    y = (df['x1'] - df['x2'] + t + rng.normal(scale=0.5, size=n)).values
    return df, t, y


def test_propensity_shap_probability_columns():
    df, t, y = make_dummy_data()
    data = CausalDataLite(df=df.assign(T=t, Y=y), treatment='T', target='Y', confounders=['x1', 'x2', 'c1'])
    eda = CausalEDA(data, ps_model=CatBoostClassifier(thread_count=-1, verbose=False, random_state=7), n_splits=3, random_state=7)

    ps_model = eda.fit_propensity()
    shap_df = ps_model.shap

    # Columns added
    for col in ['odds_mult_abs', 'exact_pp_change_abs', 'exact_pp_change_signed']:
        assert col in shap_df.columns

    # Numerical checks
    p0 = float(np.mean(ps_model.propensity_scores))
    p0 = float(np.clip(p0, 1e-9, 1 - 1e-9))
    logit_p0 = float(np.log(p0 / (1.0 - p0)))

    odds_mult_abs_exp = np.exp(shap_df['shap_mean_abs'].values)
    exact_pp_change_abs_exp = 1.0 / (1.0 + np.exp(-(logit_p0 + shap_df['shap_mean_abs'].values))) - p0
    exact_pp_change_signed_exp = 1.0 / (1.0 + np.exp(-(logit_p0 + shap_df['shap_mean'].values))) - p0

    assert np.allclose(shap_df['odds_mult_abs'].values, odds_mult_abs_exp, rtol=1e-6, atol=1e-8)
    assert np.allclose(shap_df['exact_pp_change_abs'].values, exact_pp_change_abs_exp, rtol=1e-6, atol=1e-8)
    assert np.allclose(shap_df['exact_pp_change_signed'].values, exact_pp_change_signed_exp, rtol=1e-6, atol=1e-8)


def test_eda_treatment_features_probability_columns():
    df, t, y = make_dummy_data(seed=456)
    data = CausalDataLite(df=df.assign(T=t, Y=y), treatment='T', target='Y', confounders=['x1', 'x2', 'c1'])
    eda = CausalEDA(data, ps_model=CatBoostClassifier(thread_count=-1, verbose=False, random_state=3), n_splits=3, random_state=3)

    _ = eda.fit_propensity()
    shap_df = eda.treatment_features()

    for col in ['odds_mult_abs', 'exact_pp_change_abs', 'exact_pp_change_signed']:
        assert col in shap_df.columns

    p0 = float(np.mean(eda._ps))
    p0 = float(np.clip(p0, 1e-9, 1 - 1e-9))
    logit_p0 = float(np.log(p0 / (1.0 - p0)))

    odds_mult_abs_exp = np.exp(shap_df['shap_mean_abs'].values)
    exact_pp_change_abs_exp = 1.0 / (1.0 + np.exp(-(logit_p0 + shap_df['shap_mean_abs'].values))) - p0
    exact_pp_change_signed_exp = 1.0 / (1.0 + np.exp(-(logit_p0 + shap_df['shap_mean'].values))) - p0

    assert np.allclose(shap_df['odds_mult_abs'].values, odds_mult_abs_exp, rtol=1e-6, atol=1e-8)
    assert np.allclose(shap_df['exact_pp_change_abs'].values, exact_pp_change_abs_exp, rtol=1e-6, atol=1e-8)
    assert np.allclose(shap_df['exact_pp_change_signed'].values, exact_pp_change_signed_exp, rtol=1e-6, atol=1e-8)
