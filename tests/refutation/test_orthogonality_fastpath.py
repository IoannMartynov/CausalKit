import numpy as np
import pandas as pd

from causalis.data import CausalData
from causalis.inference.ate.dml_ate import dml_ate
from causalis.refutation.score.score_validation import (
    refute_irm_orthogonality,
    oos_moment_check_from_psi,
)


def _make_data(n=300, seed=123):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    lin = 0.7 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-lin))
    d = rng.binomial(1, p)
    y = 1.5 * d + x1 - 0.5 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])


def test_fastpath_oos_equals_helper():
    # Fit IRM once to get psi components and folds
    data = _make_data(n=240, seed=777)
    res = dml_ate(data, n_folds=3, random_state=777)
    model = res["model"]

    # Build fold indices from cross-fitting folds
    folds = np.asarray(getattr(model, "folds_"))
    K = int(folds.max() + 1)
    fold_indices = [np.where(folds == k)[0] for k in range(K)]

    # Helper-computed t-stats from cached psi
    df_fast, t_fold_fast, t_strict_fast = oos_moment_check_from_psi(
        np.asarray(model.psi_a_), np.asarray(model.psi_b_), fold_indices, strict=True
    )

    # Function output should match helper (fast path is used under the hood)
    out = refute_irm_orthogonality(dml_ate, data, n_folds_oos=3, strict_oos=True)
    t_fold_fn = out["oos_moment_test"]["tstat_fold_agg"]
    t_strict_fn = out["oos_moment_test"]["tstat_strict"]

    assert abs(t_fold_fn - t_fold_fast) < 1e-10
    assert abs(t_strict_fn - t_strict_fast) < 1e-10
