from types import SimpleNamespace

from causalis.refutation.unconfoundedness.uncofoundedness_validation import get_sensitivity_summary


def test_bias_aware_summary_formatting_includes_intervals():
    # Construct a minimal effect dict with pre-populated bias-aware components
    effect = {
        'model': SimpleNamespace(data=SimpleNamespace(treatment=SimpleNamespace(name='d'))),
        'bias_aware': {
            'theta': 1.2345,
            'se': 0.1111,
            'level': 0.95,
            'z': 1.959964,
            'sampling_ci': (0.9000, 1.5000),
            'theta_bounds_confounding': (1.1000, 1.3000),
            'bias_aware_ci': (0.8000, 1.7000),
            'max_bias': 0.1000,
            'sigma2': 0.25,
            'nu2': 0.04,
            'params': {'cf_y': 0.1, 'cf_d': 0.2, 'rho': 0.3, 'use_signed_rr': False},
        },
    }

    text = get_sensitivity_summary(effect)
    assert isinstance(text, str)
    assert 'Bias-aware Interval' in text
    assert 'Intervals' in text
    # Check formatted numbers appear with 6 decimals in the intervals row
    assert '  0.900000' in text and '  1.500000' in text
    assert '  1.100000' in text and '  1.300000' in text
