import pytest
import pandas as pd

from causalis.data import CausalData
from causalis.refutation.score.score_validation import (
    refute_placebo_outcome,
    refute_placebo_treatment,
    refute_subset,
)


def _dummy_inference_fn(data: CausalData, **kwargs):
    # Minimal mock inference function returning required keys
    return {
        "coefficient": 0.0,
        "p_value": 1.0,
        # extra keys are ignored by placebo helpers
    }


def _make_causal_data():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "d": [0, 1, 0, 1, 0],
        "x1": [0.1, 0.2, 0.3, 0.4, 0.6],
        "x2": [1.0, 0.0, 1.0, 0.0, 1.0],
    })
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


@pytest.mark.parametrize("fn_name", [
    "outcome", "treatment", "subset"
])
def test_placebo_helpers_run_without_errors(fn_name):
    data = _make_causal_data()
    if fn_name == "outcome":
        res = refute_placebo_outcome(_dummy_inference_fn, data, random_state=123)
    elif fn_name == "treatment":
        res = refute_placebo_treatment(_dummy_inference_fn, data, random_state=123)
    else:
        # Use full-sample subset to avoid degenerate constant-treatment cases in tiny data
        res = refute_subset(_dummy_inference_fn, data, fraction=1.0, random_state=123)

    assert isinstance(res, dict)
    assert set(["theta", "p_value"]).issubset(res.keys())
    assert isinstance(res["theta"], float)
    assert isinstance(res["p_value"], float)
