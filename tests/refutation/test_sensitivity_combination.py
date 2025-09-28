import re
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from causalkit.data.causaldata import CausalData
from causalkit.inference.estimators.irm import IRM
from causalkit.refutation.unconfoundedness.uncofoundedness_validation import sensitivity_analysis, get_sensitivity_summary


def make_synth(n=400, seed=123):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 1.2 * x1 + 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p)
    y = 1.0 * d + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return df


def fit_irm(df):
    data = CausalData(df, treatment="d", outcome="y", confounders=["x1", "x2"])
    ml_g = RandomForestRegressor(n_estimators=60, random_state=1)
    ml_m = LogisticRegression(max_iter=1000)
    irm = IRM(data=data, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=1)
    irm.fit()
    return irm


def _parse_bounds(summary: str):
    # Find the table data line (after header). It contains 5 floating numbers right-aligned
    lines = [ln for ln in summary.splitlines() if ln.strip() and not ln.startswith("=")]
    # Find the line under the Bounds header: it's the next non-header line that starts with treatment name or spaces
    start_idx = None
    for i, ln in enumerate(lines):
        if "Bounds with CI" in ln:
            start_idx = i
            break
    assert start_idx is not None
    # The data line is two lines below the header (header row then data row)
    data_line = lines[start_idx + 2]
    # Extract floats from the data line
    floats = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", data_line)]
    # The five numbers are: CI lower, theta lower, theta, theta upper, CI upper
    # Return theta lower, theta, theta upper
    return floats[1], floats[2], floats[3]


def test_zero_confounding_collapses_bounds():
    df = make_synth()
    irm = fit_irm(df)
    effect = {"model": irm}
    summary = sensitivity_analysis(effect, cf_y=0.0, cf_d=0.0, rho=0.5, level=0.95)
    # use top-level getter to ensure it's stored
    s = get_sensitivity_summary(effect)
    assert isinstance(s, str)
    tl, th, tu = _parse_bounds(s)
    assert pytest.approx(tl, 1e-10) == th
    assert pytest.approx(tu, 1e-10) == th


def test_rho_sign_affects_bounds():
    df = make_synth()
    irm = fit_irm(df)
    effect = {"model": irm}
    s_pos = sensitivity_analysis(effect, cf_y=0.2, cf_d=0.2, rho=+1.0, level=0.95)
    s_neg = sensitivity_analysis(effect, cf_y=0.2, cf_d=0.2, rho=-1.0, level=0.95)
    tl_pos, th_pos, tu_pos = _parse_bounds(get_sensitivity_summary(effect))
    # call again for negative rho and parse freshly
    s = sensitivity_analysis(effect, cf_y=0.2, cf_d=0.2, rho=-1.0, level=0.95)
    tl_neg, th_neg, tu_neg = _parse_bounds(get_sensitivity_summary(effect))
    # Widths
    w_pos = tu_pos - th_pos
    w_neg = tu_neg - th_neg
    assert w_pos >= 0 and w_neg >= 0
    # Positive rho should widen more than negative rho
    assert w_pos >= w_neg


def test_input_validation_and_header_label():
    df = make_synth()
    irm = fit_irm(df)
    effect = {"model": irm}

    # Invalid level -> ValueError at top-level (validated before delegation)
    with pytest.raises(ValueError):
        sensitivity_analysis(effect, cf_y=0.1, cf_d=0.1, rho=0.0, level=1.0)
    with pytest.raises(ValueError):
        sensitivity_analysis(effect, cf_y=0.1, cf_d=0.1, rho=0.0, level=0.0)

    # Negative cf_y -> ValueError
    with pytest.raises(ValueError):
        sensitivity_analysis(effect, cf_y=-0.1, cf_d=0.1, rho=0.0, level=0.95)

    # Header includes SNR proxy label
    summary = sensitivity_analysis(effect, cf_y=0.05, cf_d=0.05, rho=0.0, level=0.95)
    assert "SNR proxy" in summary
