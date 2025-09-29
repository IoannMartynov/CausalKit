"""
Simple IRM-based implementation for estimating ATT (Average Treatment effect on the Treated).

This module provides a function dml_att_s to estimate ATT using our internal
DoubleML-style IRM estimator that consumes CausalData directly (not DoubleML).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from causalkit.data.causaldata import CausalData
from causalkit.inference.estimators import IRM


def dml_atte(
    data: CausalData,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 4,
    n_rep: int = 1,
    confidence_level: float = 0.95,
    normalize_ipw: bool = False,
    trimming_rule: str = "truncate",
    trimming_threshold: float = 1e-2,
    random_state: Optional[int] = None,
    store_diagnostic_data: bool = True,
) -> Dict[str, Any]:
    """
    Estimate average treatment effect on the treated (ATT) using the internal IRM estimator.

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment, outcome, and confounders.
    ml_g : estimator, optional
        Learner for g(D,X)=E[Y|X,D]. If outcome is binary and learner is classifier,
        predict_proba will be used; otherwise predict().
    ml_m : classifier, optional
        Learner for m(X)=E[D|X] (propensity). If None, a CatBoostClassifier is used.
    n_folds : int, default 5
        Number of folds for cross-fitting.
    n_rep : int, default 1
        Number of repetitions (currently only 1 supported by IRM).
    confidence_level : float, default 0.95
        Confidence level for CI in (0,1).
    normalize_ipw : bool, default False
        Whether to normalize IPW terms within the score.
    trimming_rule : str, default "truncate"
        Trimming approach for propensity (only "truncate" supported).
    trimming_threshold : float, default 1e-2
        Trimming threshold for propensity.
    random_state : int, optional
        Random seed for fold creation.

    Returns
    -------
    Dict[str, Any]
        Keys: coefficient, std_error, p_value, confidence_interval, model
    
    Notes
    -----
    By default, this function stores a comprehensive 'diagnostic_data' dictionary in the result.
    You can disable this by setting store_diagnostic_data=False.
    """
    # Basic validations similar to existing wrappers
    if data.treatment is None:
        raise ValueError("CausalData object must have a treatment variable defined")
    if data.target is None:
        raise ValueError("CausalData object must have a outcome variable defined")
    if not data.confounders:
        raise ValueError("CausalData object must have confounders variables defined")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")

    # Defaults for learners: lazy import CatBoost only if needed
    if ml_g is None or ml_m is None:
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier  # type: ignore
        except ImportError as e:
            raise ImportError(
                "CatBoost is required for default learners. Install 'catboost' or provide ml_g and ml_m."
            ) from e
        if ml_g is None:
            ml_g = CatBoostRegressor(
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
            )
        if ml_m is None:
            ml_m = CatBoostClassifier(
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
            )

    # Normalize treatment to 0/1 if boolean to keep CausalData consistent for IRM
    df = data.get_df().copy()
    tname = data.treatment.name
    if df[tname].dtype == bool:
        df[tname] = df[tname].astype(int)
        data = CausalData(df=df, treatment=tname, outcome=data.target.name, confounders=data.confounders)
    else:
        uniq = np.unique(df[tname].values)
        if not np.array_equal(np.sort(uniq), np.array([0, 1])) and not np.array_equal(np.sort(uniq), np.array([0.0, 1.0])):
            raise ValueError(f"Treatment must be binary 0/1 or boolean; found {uniq}.")

    # Fit IRM with ATT (ATTE score)
    irm = IRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=n_folds, n_rep=n_rep, score="ATTE", normalize_ipw=normalize_ipw,
              trimming_rule=trimming_rule, trimming_threshold=trimming_threshold, random_state=random_state)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        irm.fit()

    # Confidence interval
    ci_df = irm.confint(level=confidence_level)
    if isinstance(ci_df, pd.DataFrame):
        ci_lower = float(ci_df.iloc[0, 0])
        ci_upper = float(ci_df.iloc[0, 1])
    else:
        arr = np.asarray(ci_df)
        ci_lower = float(arr[0, 0])
        ci_upper = float(arr[0, 1])

    # Collect diagnostic data for overlap/weight and score checks (optional)
    diagnostic_data = None
    if store_diagnostic_data:
        df_diag = data.get_df()
        y_diag = df_diag[data.target.name].to_numpy(dtype=float)
        d_diag = df_diag[data.treatment.name].to_numpy().astype(int)
        x_diag = df_diag[data.confounders].to_numpy(dtype=float)
        p1_diag = float(np.mean(d_diag))

        diagnostic_data = {
            "m_hat": np.asarray(irm.m_hat_, dtype=float),
            "g0_hat": np.asarray(irm.g0_hat_, dtype=float),
            "g1_hat": np.asarray(irm.g1_hat_, dtype=float),
            "y": y_diag,
            "d": d_diag,
            "x": x_diag,
            "psi": np.asarray(irm.psi_, dtype=float),
            "psi_a": np.asarray(irm.psi_a_, dtype=float),
            "psi_b": np.asarray(irm.psi_b_, dtype=float),
            "folds": np.asarray(getattr(irm, "folds_", None), dtype=int) if getattr(irm, "folds_", None) is not None else None,
            "score": "ATTE",
            "normalize_ipw": bool(normalize_ipw),
            "trimming_threshold": float(trimming_threshold),
            "p1": p1_diag,
        }

    return {
        "coefficient": float(irm.coef[0]),
        "std_error": float(irm.se[0]),
        "p_value": float(irm.pvalues[0]),
        "confidence_interval": (ci_lower, ci_upper),
        "model": irm,
        "diagnostic_data": diagnostic_data,
    }
