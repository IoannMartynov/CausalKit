import pandas as pd

from causalkit.refutation.unconfoundedness.uncofoundedness_validation import _format_sensitivity_summary


def test_format_sensitivity_summary_accepts_spaced_ci_columns():
    # Build a one-row DataFrame mimicking DoubleML style with spaced CI labels
    df = pd.DataFrame({
        'theta': [1.2345],
        'theta lower': [1.1000],
        'theta upper': [1.3000],
        '2.5 %': [0.9000],
        '97.5 %': [1.5000],
    }, index=['d'])

    text = _format_sensitivity_summary(df, cf_y=0.1, cf_d=0.2, rho=0.3, level=0.95)

    assert 'Bounds with CI' in text
    # Values should be formatted with 6 decimals in the assembled row
    assert '  0.900000' in text and '  1.500000' in text
    assert '  1.100000' in text and '  1.300000' in text
