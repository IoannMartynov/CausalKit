import pytest
import numpy as np
from causalkit.data import CausalDatasetGenerator


def test_generate_no_runtime_warnings_large_n():
    """
    Ensure generate() does not emit numerical RuntimeWarnings (overflow/divide/invalid)
    for a realistic configuration similar to library examples.
    """
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

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=RuntimeWarning)
        df = gen.generate(10_000)
    # Debug: show any runtime warnings captured
    for wi in w:
        if isinstance(wi.message, RuntimeWarning):
            print("RuntimeWarning:", wi.message, "at", getattr(wi, 'filename', ''), getattr(wi, 'lineno', ''))
    # Ensure no RuntimeWarnings captured
    assert not any(isinstance(wi.message, RuntimeWarning) for wi in w)

    # Basic sanity on outputs
    assert set(df.columns) >= {"y", "t", "age", "smoker", "bmi", "propensity", "mu0", "mu1", "cate"}
    assert np.isfinite(df[["y", "t", "propensity", "mu0", "mu1", "cate"]]).all().all()
