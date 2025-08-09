import numpy as np
import pandas as pd
import pytest

from causalkit.data import generate_rct_data


@pytest.mark.parametrize(
    "target_type,params",
    [
        ("binary", {"p": {"A": 0.10, "B": 0.20}}),
        ("normal", {"mean": {"A": 0.0, "B": 1.0}, "std": 2.0}),
        ("nonnormal", {"shape": 2.0, "scale": {"A": 1.0, "B": 1.3}}),
    ],
)
def test_generate_rct_data_split_and_outcome(target_type, params):
    n = 5000
    split = 0.3
    df = generate_rct_data(n_users=n, split=split, random_state=123, target_type=target_type, target_params=params)

    # Basic columns
    expected_cols = {
        "user_id","treatment","outcome","age","cnt_trans","platform_Android","platform_iOS","invited_friend"
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == n

    # Split check
    t_rate = df["treatment"].mean()
    assert abs(t_rate - split) < 0.03

    # Outcome checks by treatment group
    if target_type == "binary":
        grp_rates = df.groupby("treatment")["outcome"].mean()
        # Expect B (1) close to pB and A (0) close to pA
        pA, pB = params["p"]["A"], params["p"]["B"]
        assert abs(grp_rates.loc[0] - pA) < 0.03
        assert abs(grp_rates.loc[1] - pB) < 0.03
    elif target_type == "normal":
        grp_means = df.groupby("treatment")["outcome"].mean()
        muA, muB = params["mean"]["A"], params["mean"]["B"]
        # allow looser tolerance due to sampling noise
        assert abs(grp_means.loc[0] - muA) < 0.15
        assert abs(grp_means.loc[1] - muB) < 0.15
    else:  # nonnormal -> Poisson approx; check mean relation order
        grp_means = df.groupby("treatment")["outcome"].mean()
        shape = params.get("shape", 2.0)
        lamA = shape * params["scale"]["A"]
        lamB = shape * params["scale"]["B"]
        # Means should be close in order and roughly match
        assert grp_means.loc[1] > grp_means.loc[0]
        assert abs(grp_means.loc[0] - lamA) / max(lamA, 1e-6) < 0.25
        assert abs(grp_means.loc[1] - lamB) / max(lamB, 1e-6) < 0.25
