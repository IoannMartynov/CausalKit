"""
DoubleML implementation for estimating CATE (per-observation orthogonal signals).

This module provides a function that, given a CausalData object, fits a DoubleML IRM
model and augments the data with a new column 'cate' that contains the orthogonal
signals (an estimate of the conditional average treatment effect for each unit).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

import doubleml as dml
from catboost import CatBoostRegressor, CatBoostClassifier

from causalkit.data.causaldata import CausalData


def cate_esimand(
    data: CausalData,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    use_blp: bool = False,
    X_new: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

    Parameters
    ----------
    data : CausalData
        A CausalData object with defined target (outcome), treatment (binary 0/1), and confounders.
    ml_g : estimator, optional
        ML learner for outcome regression g(D, X) = E[Y | D, X] supporting fit/predict.
        Defaults to CatBoostRegressor if None.
    ml_m : classifier, optional
        ML learner for propensity m(X) = P[D=1 | X] supporting fit/predict_proba.
        Defaults to CatBoostClassifier if None.
    n_folds : int, default 5
        Number of folds for cross-fitting.
    n_rep : int, default 1
        Number of repetitions for sample splitting.
    use_blp : bool, default False
        If True, and X_new is provided, returns cate from obj.blp_predict(X_new) aligned to X_new.
        If False (default), uses obj._orthogonal_signals (in-sample estimates) and appends to data.
    X_new : pd.DataFrame, optional
        New covariate matrix for out-of-sample CATE prediction via best linear predictor.
        Must contain the same feature columns as the confounders in `data`.

    Returns
    -------
    pd.DataFrame
        If use_blp is False: returns a copy of data.df with a new column 'cate'.
        If use_blp is True and X_new is provided: returns a DataFrame with 'cate' column for X_new rows.

    Raises
    ------
    ValueError
        If treatment is not binary 0/1 or required metadata is missing.
    """
    # Basic validation similar to ATE/ATT implementations for consistency
    if data.treatment is None:
        raise ValueError("CausalData object must have a treatment variable defined")
    if data.target is None:
        raise ValueError("CausalData object must have a target variable defined")
    if data.cofounders is None:
        raise ValueError("CausalData object must have cofounders variables defined")

    # Check binary treatment with explicit {0,1}
    unique_treatments = pd.Series(data.treatment).unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")
    if set(unique_treatments) != {0, 1}:
        raise ValueError("Treatment variable must have values 0 and 1")

    # Defaults for learners
    if ml_g is None:
        ml_g = CatBoostRegressor(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)
    if ml_m is None:
        ml_m = CatBoostClassifier(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)

    # Data for DoubleML
    df = data.get_df()
    dml_data = dml.DoubleMLData(
        df,
        y_col=data._target,
        d_cols=data._treatment,
        x_cols=data._cofounders,
    )

    # Fit DoubleML IRM
    obj = dml.DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score="ATE",
    ).fit()

    if use_blp:
        # Out-of-sample via best linear predictor; require X_new
        if X_new is None:
            raise ValueError("X_new must be provided when use_blp=True")
        # Ensure columns match the confounders order
        X_cols = list(data._cofounders)
        missing = [c for c in X_cols if c not in X_new.columns]
        if missing:
            raise ValueError(f"X_new is missing required columns: {missing}")
        # Some DoubleML versions may not expose blp_predict
        if not hasattr(obj, "blp_predict"):
            raise NotImplementedError("This DoubleML version does not support blp_predict. Use use_blp=False for in-sample CATEs.")
        cate_hat = obj.blp_predict(X_new[X_cols])
        # Return DataFrame aligned with X_new
        return pd.DataFrame({"cate": np.asarray(cate_hat).reshape(-1)})

    # In-sample orthogonal signals as CATE estimates
    if hasattr(obj, "_orthogonal_signals"):
        cate_hat = np.asarray(obj._orthogonal_signals).reshape(-1)
    elif hasattr(obj, "psi_elements") and isinstance(obj.psi_elements, dict) and "psi_b" in obj.psi_elements:
        psi_b = obj.psi_elements["psi_b"]
        arr = np.asarray(psi_b)
        # Average across all axes except the first to obtain one value per observation
        if arr.ndim == 1:
            cate_hat = arr
        else:
            axes = tuple(range(1, arr.ndim))
            cate_hat = np.nanmean(arr, axis=axes)
    else:
        raise AttributeError("Could not extract orthogonal signals from DoubleMLIRM object.")

    out = df.copy()
    out["cate"] = np.asarray(cate_hat).reshape(-1)
    return out
