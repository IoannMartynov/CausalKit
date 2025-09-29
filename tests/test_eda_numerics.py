import warnings
import numpy as np
import pandas as pd

from causalkit.data import CausalDatasetGenerator, CausalData
from causalkit.eda import CausalEDA




def test_fit_propensity_produces_valid_ps():
    # smaller sample for speed
    gen = CausalDatasetGenerator(seed=123, confounder_specs=[
        {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
        {"name": "x2", "dist": "bernoulli", "p": 0.4},
    ], target_t_rate=0.3)
    df = gen.generate(1000)
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])
    eda = CausalEDA(cd)

    ps_model = eda.fit_propensity()
    ps = ps_model.propensity_scores
    assert ps.shape == (1000,)
    assert np.all(np.isfinite(ps))
    assert np.all((ps > 0) & (ps < 1))
