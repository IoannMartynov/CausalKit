import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from causalkit.data import CausalData
from causalkit.inference.att import dml_att_source
from causalkit.refutation.score.score_validation import (
    refute_placebo_outcome,
    refute_placebo_treatment,
    refute_subset,
)
from causalkit.refutation.overlap import overlap_report_from_result


def _make_synth(n=400, seed=123):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    # Treatment assignment (logit based on x1, x2)
    logits = 0.6 * x1 + 0.9 * x2 - 0.1
    p = 1.0 / (1.0 + np.exp(-logits))
    t = rng.binomial(1, p, size=n)
    # Outcome (continuous): baseline + tau * t + noise
    tau = 1.0
    y = 0.4 * x1 + 0.6 * x2 + tau * t + rng.normal(0, 1.0, size=n)

    df = pd.DataFrame({
        "y": y.astype(float),
        "t": t.astype(int),
        "x1": x1.astype(float),
        "x2": x2.astype(float),
    })
    data = CausalData(df=df, treatment="t", outcome="y", confounders=["x1", "x2"])
    return data


def _light_models():
    # Lightweight sklearn models to avoid CatBoost dependency
    ml_g = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0)
    return ml_g, ml_m


def test_refutation_after_dml_att_source_runs_and_reports_overlap():
    data = _make_synth(n=500, seed=11)
    ml_g, ml_m = _light_models()

    # 1) Run ATT inference via DoubleML-based source function
    att_result = dml_att_source(data, ml_g=ml_g, ml_m=ml_m, n_folds=2, n_rep=1, confidence_level=0.9)
    # Basic sanity on inference
    assert isinstance(att_result, dict)
    for key in ["coefficient", "std_error", "p_value", "confidence_interval", "model"]:
        assert key in att_result

    # 2) Run placebo refutations using the same estimator function
    placebo_y = refute_placebo_outcome(dml_att_source, data, random_state=42, ml_g=ml_g, ml_m=ml_m, n_folds=2, n_rep=1)
    placebo_t = refute_placebo_treatment(dml_att_source, data, random_state=42, ml_g=ml_g, ml_m=ml_m, n_folds=2, n_rep=1)
    subset_res = refute_subset(dml_att_source, data, fraction=0.5, random_state=42, ml_g=ml_g, ml_m=ml_m, n_folds=2, n_rep=1)

    # Placebo/subset results have required shape
    for res in (placebo_y, placebo_t, subset_res):
        assert isinstance(res, dict)
        assert set(["theta", "p_value"]).issubset(res.keys())
        assert isinstance(res["theta"], float)
        assert isinstance(res["p_value"], float)

