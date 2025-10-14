import numpy as np
import pandas as pd

from causalis.data import CausalData
from causalis.inference.ate.dml_ate import dml_ate
from causalis.refutation.score.score_validation import trim_sensitivity_curve_ate


def _make_data(n=300, seed=321):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    lin = 0.6 * x1 - 0.4 * x2
    p = 1.0 / (1.0 + np.exp(-lin))
    d = rng.binomial(1, p)
    y = 2.0 * d + x1 - 0.5 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


def test_trim_sensitivity_curve_ate_basic_and_matches_model_at_default_eps():
    data = _make_data(n=240, seed=777)
    # Fit with default settings (ATE, normalize_ipw=False, trimming_threshold=1e-2)
    res = dml_ate(data, n_folds=3, random_state=777)
    model = res["model"]
    dd = res["diagnostic_data"]

    m = dd["m_hat"]
    d = dd["d"]
    y = dd["y"]
    g0 = dd["g0_hat"]
    g1 = dd["g1_hat"]
    eps0 = float(dd.get("trimming_threshold", 0.01))

    grid = (0.0, eps0, 0.02)
    df = trim_sensitivity_curve_ate(m, d, y, g0, g1, eps_grid=grid)

    # Columns and length
    assert isinstance(df, pd.DataFrame)
    assert set(["trim_eps", "n", "pct_clipped", "theta", "se"]).issubset(df.columns)
    assert len(df) == len(grid)

    # Monotone non-decreasing pct_clipped as eps increases
    df_sorted = df.sort_values("trim_eps")
    pct = df_sorted["pct_clipped"].to_numpy()
    assert np.all(np.diff(pct) >= -1e-12)

    # At eps equal to the model's trimming threshold, theta should match the model's coefficient
    theta_at_eps0 = float(df.loc[np.isclose(df["trim_eps"], eps0), "theta"].iloc[0])
    assert abs(theta_at_eps0 - float(res["coefficient"])) < 1e-8
