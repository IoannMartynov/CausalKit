import numpy as np
import pandas as pd
import pytest

from causalkit.data import generate_rct


@pytest.mark.parametrize(
    "target_type,params",
    [
        ("binary", {"p": {"A": 0.10, "B": 0.20}}),
        ("normal", {"mean": {"A": 0.0, "B": 1.0}, "std": 2.0}),
        ("nonnormal", {"shape": 2.0, "scale": {"A": 1.0, "B": 1.3}}),
    ],
)
def test_generate_rct_split_and_outcome(target_type, params):
    n = 5000
    split = 0.3
    df = generate_rct(n=n, split=split, random_state=123, target_type=target_type, target_params=params, add_ancillary=False)

    # Basic columns for new schema
    required_cols = {"y", "t", "propensity", "mu0", "mu1", "cate"}
    assert required_cols.issubset(set(df.columns))
    assert len(df) == n

    # Split check
    t_rate = df["t"].mean()
    assert abs(t_rate - split) < 0.03

    # Outcome checks by treatment group
    if target_type == "binary":
        grp_rates = df.groupby("t")["y"].mean()
        pA, pB = params["p"]["A"], params["p"]["B"]
        assert abs(grp_rates.loc[0.0] - pA) < 0.03
        assert abs(grp_rates.loc[1.0] - pB) < 0.03
    elif target_type == "normal":
        grp_means = df.groupby("t")["y"].mean()
        muA, muB = params["mean"]["A"], params["mean"]["B"]
        assert abs(grp_means.loc[0.0] - muA) < 0.15
        assert abs(grp_means.loc[1.0] - muB) < 0.15
    else:  # nonnormal -> Poisson alias -> check means near lamA, lamB
        grp_means = df.groupby("t")["y"].mean()
        shape = params.get("shape", 2.0)
        lamA = shape * params["scale"]["A"]
        lamB = shape * params["scale"]["B"]
        assert grp_means.loc[1.0] > grp_means.loc[0.0]
        assert abs(grp_means.loc[0.0] - lamA) / max(lamA, 1e-6) < 0.25
        assert abs(grp_means.loc[1.0] - lamB) / max(lamB, 1e-6) < 0.25
