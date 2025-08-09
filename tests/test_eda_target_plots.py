import matplotlib
matplotlib.use('Agg')  # non-interactive backend for tests

import numpy as np
import pandas as pd
from causalkit.data import CausalDatasetGenerator, CausalData
from causalkit.eda import CausalEDA


def test_plot_target_by_treatment_continuous():
    gen = CausalDatasetGenerator(
        theta=1.5,
        beta_y=np.array([0.5, -0.2, 0.1]),
        beta_t=np.array([0.3, 0.7, -0.1]),
        target_t_rate=0.4,
        outcome_type="continuous",
        sigma_y=1.0,
        seed=123,
        confounder_specs=[
            {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
            {"name": "x2", "dist": "bernoulli", "p": 0.4},
            {"name": "x3", "dist": "normal", "mu": 2, "sd": 0.5},
        ],
    )
    df = gen.generate(1000)
    cd = CausalData(df=df, treatment="t", outcome="y", cofounders=["x1", "x2", "x3"])
    eda = CausalEDA(cd)

    fig1, fig2 = eda.plot_target_by_treatment()
    # Basic assertions: figures created
    from matplotlib.figure import Figure
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
