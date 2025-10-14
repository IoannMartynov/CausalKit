import numpy as np
import pandas as pd
from types import SimpleNamespace

from causalis.refutation.unconfoundedness.uncofoundedness_validation import (
    sensitivity_analysis,
    get_sensitivity_summary,
)


class DummyData:
    def __init__(self):
        self.treatment = SimpleNamespace(name='D')
        self.target = SimpleNamespace(name='Y')
        self.confounders = ['x1']

    def get_df(self):
        # minimal DF not used by sensitivity functions
        return pd.DataFrame({"D": [0, 1], "Y": [0.0, 1.0], "x1": [0.0, 1.0]})


class DummyModel:
    def __init__(self, rr):
        self.data = DummyData()
        self.coef_ = np.array([1.0])
        self.se_ = np.array([0.2])
        self._rr = np.asarray(rr, dtype=float)

    def confint(self, level=0.95):
        from scipy.stats import norm
        z = norm.ppf(0.5 + level/2.0)
        theta = float(self.coef_[0])
        se = float(self.se_[0])
        return pd.DataFrame({
            '2.5 %': [theta - z*se],
            '97.5 %': [theta + z*se],
        })

    def _sensitivity_element_est(self):
        n = self._rr.size
        return {
            'sigma2': 0.25,
            'psi_sigma2': np.zeros(n, dtype=float),
            'm_alpha': np.full(n, 0.5, dtype=float),
            'riesz_rep': self._rr.copy(),
        }


def make_result(rr):
    return {
        'model': DummyModel(rr),
        # No need to pass coefficient/std_error/CI; puller uses model
    }


def test_bias_aware_summary_basic():
    res = make_result(rr=[1.0, -1.0, 0.0, 0.5, -0.2])
    out = sensitivity_analysis(res, cf_y=0.01, cf_d=0.01, rho=1.0, level=0.95)
    assert isinstance(out, dict)
    # bias-aware dict must contain components
    assert 'bias_aware_ci' in out and 'theta_bounds_confounding' in out
    # get_sensitivity_summary should render a bias-aware interval string
    summary2 = get_sensitivity_summary(res)
    assert isinstance(summary2, str)
    assert "Bias-aware Interval" in summary2
    # bias-aware is also cached on the effect_estimation
    ba = res.get('bias_aware')
    assert isinstance(ba, dict)
    assert 'bias_aware_ci' in ba and 'theta_bounds_confounding' in ba
    # Bias-aware CI should be wider than sampling CI
    ci_l, ci_u = ba['sampling_ci']
    bci_l, bci_u = ba['bias_aware_ci']
    assert bci_l <= ci_l and bci_u >= ci_u


def test_use_signed_rr_flag_effect():
    res1 = make_result(rr=[1.0, -1.0, 0.0, 0.5, -0.2])
    res2 = make_result(rr=[1.0, -1.0, 0.0, 0.5, -0.2])
    ba1 = sensitivity_analysis(res1, cf_y=0.01, cf_d=0.02, rho=1.0, level=0.95, use_signed_rr=False)
    ba2 = sensitivity_analysis(res2, cf_y=0.01, cf_d=0.02, rho=1.0, level=0.95, use_signed_rr=True)
    # Worst-case (abs rr) should not be smaller than signed rr case
    assert ba1['max_bias'] >= ba2['max_bias'] - 1e-12
