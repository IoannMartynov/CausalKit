import warnings
import numpy as np
import pandas as pd

from causalkit.data import CausalDatasetGenerator, CausalData
from causalkit.eda import CausalEDA


def test_design_report_no_runtime_warnings():
    gen = CausalDatasetGenerator(
        theta=2.0,
        beta_y=np.array([1.0, -0.5, 0.2]),
        beta_t=np.array([0.8, 1.2, -0.3]),
        target_t_rate=0.35,
        outcome_type="continuous",
        sigma_y=1.0,
        seed=42,
        confounder_specs=[
            {"name": "age", "dist": "normal", "mu": 50, "sd": 10},
            {"name": "smoker", "dist": "bernoulli", "p": 0.3},
            {"name": "bmi", "dist": "normal", "mu": 27, "sd": 4},
        ],
    )
    df = gen.generate(4000)
    cd = CausalData(df=df, treatment="t", outcome="y", confounders=["age", "smoker", "bmi"])

    eda = CausalEDA(cd)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=RuntimeWarning)
        report = eda.design_report()
    # Ensure no RuntimeWarnings captured
    assert not any(isinstance(wi.message, RuntimeWarning) for wi in w)

    # Sanity checks on report keys
    assert set(report.keys()) == {"health", "summaries", "treat_auc", "positivity", "balance", "weights"}
    assert 0 <= report["treat_auc"] <= 1


def test_fit_propensity_produces_valid_ps():
    # smaller sample for speed
    gen = CausalDatasetGenerator(seed=123, confounder_specs=[
        {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
        {"name": "x2", "dist": "bernoulli", "p": 0.4},
    ], target_t_rate=0.3)
    df = gen.generate(1000)
    cd = CausalData(df=df, treatment="t", outcome="y", confounders=["x1", "x2"])
    eda = CausalEDA(cd)

    ps = eda.fit_propensity()
    assert ps.shape == (1000,)
    assert np.all(np.isfinite(ps))
    assert np.all((ps > 0) & (ps < 1))
