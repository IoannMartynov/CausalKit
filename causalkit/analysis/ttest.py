"""
Implementation of t-test and related statistical methods for A/B testing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def compare_ab(control: np.ndarray, treatment: np.ndarray):
    """
    Compares A/B test results using:
      1) two‐sample t‐test
      2) OLS regression with a treatment dummy

    Parameters
    ----------
    control : np.ndarray
        Array of values from the control group
    treatment : np.ndarray
        Array of values from the treatment group

    Returns
    -------
    None
        Prints effect estimates, test statistics, and p‐values.
    """
    # --- Two‐sample t‐test ---
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=True)
    diff = treatment.mean() - control.mean()

    # --- OLS regression ---
    df = pd.DataFrame({
        'y': np.concatenate([control, treatment]),
        'treatment': np.concatenate([np.zeros_like(control), np.ones_like(treatment)])
    })
    X = sm.add_constant(df['treatment'])
    model = sm.OLS(df['y'], X).fit()
    reg_coef = model.params['treatment']
    reg_pval = model.pvalues['treatment']

    # --- Print results ---
    print("=== Two‐sample t‐test ===")
    print(f"Estimate (μ₁–μ₀): {diff:.4f}")
    print(f"t‐statistic:         {t_stat:.4f}")
    print(f"p‐value:             {p_val:.4e}\n")

    print("=== OLS Regression ===")
    print(model.summary())


def compare_ab_with_plr(control: np.ndarray, treatment: np.ndarray):
    """
    Compares A/B test results using:
      1) two-sample t-test
      2) OLS regression with a treatment dummy
      3) PLR from DoubleML (Partial Linear Regression)

    Parameters
    ----------
    control : np.ndarray
        Array of values from the control group
    treatment : np.ndarray
        Array of values from the treatment group

    Returns
    -------
    None
        Prints effect estimates, test statistics, and p‐values.
    """
    # --- Two-sample t-test ---
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=True)
    diff = treatment.mean() - control.mean()

    # --- OLS regression ---
    df = pd.DataFrame({
        'y': np.concatenate([control, treatment]),
        'treatment': np.concatenate([np.zeros_like(control), np.ones_like(treatment)])
    })
    X_ols = sm.add_constant(df['treatment'])
    model = sm.OLS(df['y'], X_ols).fit()
    reg_coef = model.params['treatment']
    reg_pval = model.pvalues['treatment']

    # --- DoubleML PLR ---
    # Add a dummy covariate 'age' for example
    np.random.seed(42)
    df['age'] = np.random.normal(loc=40, scale=10, size=df.shape[0])

    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor

    # X - covariates (can be multiple)
    X = df[['age']]
    y = df['y']
    treatment_var = df['treatment']

    data = DoubleMLData.from_arrays(x=X, y=y, d=treatment_var)

    ml_g = RandomForestRegressor()
    ml_m = RandomForestRegressor()

    plr = DoubleMLPLR(data, ml_g, ml_m)
    plr.fit()

    # ---- Print results ----
    print("=== Two‐sample t‐test ===")
    print(f"Estimate (μ₁–μ₀): {diff:.4f}")
    print(f"t‐statistic:      {t_stat:.4f}")
    print(f"p‐value:          {p_val:.4e}\n")

    print("=== OLS Regression ===")
    print(model.summary())

    print("=== DoubleML PLR ===")
    print(plr.summary)