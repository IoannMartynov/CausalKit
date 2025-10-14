import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.data.causaldata import CausalData
from causalis.data.generators import generate_rct
from causalis.inference.ate.dml_ate import dml_ate
from causalis.refutation.unconfoundedness.uncofoundedness_validation import validate_unconfoundedness_balance


def test_unconfoundedness_balance_extras_outputs():
    df = generate_rct(n=1500, k=4, random_state=17, target_type="binary")
    confs = [c for c in df.columns if c.startswith("x")]
    data = CausalData(df=df, treatment='d', outcome='y', confounders=confs)

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=400)

    res = dml_ate(
        data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        confidence_level=0.90,
        normalize_ipw=True,
        trimming_threshold=1e-3,
        random_state=11,
        store_diagnostic_data=True,
    )

    out = validate_unconfoundedness_balance(res)
    # New fields should be present and of correct types
    assert 'smd_max' in out
    assert 'worst_features' in out
    assert np.isfinite(float(out['smd_max'])) or np.isnan(float(out['smd_max']))

    worst = out['worst_features']
    assert isinstance(worst, pd.Series)
    # worst_features index must be subset of confounders
    assert set(worst.index).issubset(set(confs))
    # Values non-negative
    assert (worst.values >= 0).all()
