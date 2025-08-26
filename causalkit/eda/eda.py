"""EDA utilities for causal analysis (propensity, overlap, balance, weights).

This module provides a lightweight CausalEDA class to quickly assess whether a
binary treatment problem is suitable for causal effect estimation. The outputs
focus on interpretability: treatment predictability, overlap/positivity,
covariate balance before/after weighting, and basic data health.

What the main outputs mean

- outcome_stats(): DataFrame with comprehensive statistics (count, mean, std, 
  percentiles, min/max) for outcome grouped by treatment.
- fit_propensity(): Numpy array of cross-validated propensity scores P(T=1|X).
- confounders_roc_auc(): Float ROC AUC of treatment vs. propensity score.
  Higher AUC implies treatment is predictable from confounders (more confounding risk).
- positivity_check(): Dict with bounds, share_below, share_above, and flag.
  It reports what share of units have PS outside [low, high]; a large share
  signals poor overlap (violated positivity).
- plot_ps_overlap(): Overlaid histograms of PS for treated vs control.
- confounders_means(): DataFrame with comprehensive balance assessment including
  means by treatment group, absolute differences, and standardized mean differences (SMD).

Note: The class accepts either the project’s CausalData object (duck-typed) or a
CausalDataLite with explicit fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt


class PropensityModel:
    """A model for propensity scores and related diagnostics.
    
    This class encapsulates propensity scores and provides methods for:
    - Computing ROC AUC
    - Extracting SHAP values
    - Plotting propensity score overlap
    - Checking positivity/overlap
    
    The class is returned by CausalEDA.fit_propensity() and provides a cleaner
    interface for propensity score analysis.
    """
    
    def __init__(self, 
                 propensity_scores: np.ndarray,
                 treatment_values: np.ndarray,
                 fitted_model: Any,
                 feature_names: List[str],
                 X_for_shap: Optional[np.ndarray] = None,
                 cat_features_for_shap: Optional[List[int]] = None):
        """Initialize PropensityModel with fitted model artifacts.
        
        Parameters
        ----------
        propensity_scores : np.ndarray
            Array of propensity scores P(T=1|X)
        treatment_values : np.ndarray
            Array of actual treatment assignments (0/1)
        fitted_model : Any
            The fitted propensity score model
        feature_names : List[str]
            Names of features used in the model
        X_for_shap : Optional[np.ndarray]
            Preprocessed feature matrix for SHAP computation
        cat_features_for_shap : Optional[List[int]]
            Categorical feature indices for SHAP computation
        """
        self.propensity_scores = propensity_scores
        self.treatment_values = treatment_values
        self.fitted_model = fitted_model
        self.feature_names = feature_names
        self.X_for_shap = X_for_shap
        self.cat_features_for_shap = cat_features_for_shap
    
    @property
    def roc_auc(self) -> float:
        """Compute ROC AUC of treatment assignment vs. propensity scores.
        
        Higher AUC means treatment is more predictable from confounders,
        indicating stronger systematic differences between groups (potential
        confounding). Values near 0.5 suggest random-like assignment.
        
        Returns
        -------
        float
            ROC AUC score between 0 and 1
        """
        return float(roc_auc_score(self.treatment_values, self.propensity_scores))
    
    @property
    def shap(self) -> pd.DataFrame:
        """Return SHAP values from the fitted propensity score model.
        
        SHAP values show the directional contribution of each feature to 
        treatment assignment prediction, where positive values increase 
        treatment probability and negative values decrease it.
        
        Returns
        -------
        pd.DataFrame
            For CatBoost models: DataFrame with columns 'feature' and 'shap_mean',
            where 'shap_mean' represents the mean SHAP value across all samples.
            
            For sklearn models: DataFrame with columns 'feature' and 'importance'
            (absolute coefficient values, for backward compatibility).
            
        Raises
        ------
        RuntimeError
            If the fitted model does not support SHAP values extraction.
        """
        # Extract SHAP values or feature importance based on model type
        if isinstance(self.fitted_model, CatBoostClassifier):
            # Use CatBoost's SHAP values for directional feature contributions
            try:
                # Import Pool for SHAP computation
                from catboost import Pool
                
                # Check if we have the required data for SHAP computation
                if self.X_for_shap is None:
                    raise RuntimeError("Preprocessed data for SHAP computation not available.")
                
                # Create Pool object for SHAP computation (numeric-only features after preprocessing)
                shap_pool = Pool(data=self.X_for_shap)
                
                # Get SHAP values - returns array of shape (n_samples, n_features + 1) 
                # where the last column is the bias term
                shap_values = self.fitted_model.get_feature_importance(type='ShapValues', data=shap_pool)
                
                # Remove bias term (last column) and compute mean SHAP values across samples
                # This gives us the average directional contribution of each feature
                shap_values_no_bias = shap_values[:, :-1]  # Remove bias column
                importance_values = np.mean(shap_values_no_bias, axis=0)  # Mean across samples
                
                feature_names = self.feature_names
                column_name = 'shap_mean'  # Use different column name to indicate SHAP values
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract SHAP values from CatBoost model: {e}")
        
        elif hasattr(self.fitted_model, 'named_steps') and hasattr(self.fitted_model.named_steps.get('clf'), 'coef_'):
            # Handle sklearn pipeline with logistic regression
            try:
                clf = self.fitted_model.named_steps['clf']
                # For logistic regression, use absolute coefficients as importance
                importance_values = np.abs(clf.coef_[0])
                
                # Need to map back to original feature names through preprocessing
                # For simplicity, if we have preprocessed features, we'll use the original feature names
                # and aggregate importance for one-hot encoded categorical features
                prep = self.fitted_model.named_steps.get('prep')
                if hasattr(prep, 'get_feature_names_out'):
                    try:
                        # Try to get feature names from preprocessor
                        transformed_names = prep.get_feature_names_out()
                        feature_names = [str(name) for name in transformed_names]
                    except:
                        # Fallback to original feature names if transformation fails
                        feature_names = self.feature_names
                        if len(importance_values) != len(feature_names):
                            # If lengths don't match due to one-hot encoding, we can't map back easily
                            # Just use indices as feature names
                            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                else:
                    feature_names = self.feature_names
                
                column_name = 'importance'  # Keep backward compatibility for sklearn models
                    
            except Exception as e:
                raise RuntimeError(f"Failed to extract feature importance from sklearn model: {e}")
        
        else:
            raise RuntimeError(f"Feature importance extraction not supported for model type: {type(self.fitted_model)}")
        
        # Ensure we have matching lengths
        if len(importance_values) != len(feature_names):
            # This can happen with preprocessing transformations
            # Create generic feature names if needed
            if len(importance_values) > len(feature_names):
                feature_names = feature_names + [f"transformed_feature_{i}" for i in range(len(feature_names), len(importance_values))]
            else:
                feature_names = feature_names[:len(importance_values)]
        
        # Create DataFrame with appropriate column name
        result_df = pd.DataFrame({
            'feature': feature_names,
            column_name: importance_values
        })
        
        # Sort appropriately: by absolute value for SHAP values (to show most impactful features),
        # by value for regular importance (higher is better)
        if column_name == 'shap_mean':
            # For SHAP values, sort by absolute value to show most impactful features first
            result_df = result_df.reindex(result_df[column_name].abs().sort_values(ascending=False).index)
        else:
            # For regular importance, sort by value (descending)
            result_df = result_df.sort_values(column_name, ascending=False)
        
        result_df = result_df.reset_index(drop=True)
        return result_df
    
    def ps_graph(self):
        """Plot overlaid histograms of propensity scores for treated vs control.
        
        Useful to visually assess group overlap. Does not return data; it draws
        on the current matplotlib figure.
        """
        plt.figure()
        t = self.treatment_values
        ps = self.propensity_scores
        plt.hist(ps[t == 1], bins=30, alpha=0.5, density=True, label="treated")
        plt.hist(ps[t == 0], bins=30, alpha=0.5, density=True, label="control")
        plt.xlabel("Propensity score")
        plt.ylabel("Density")
        plt.legend()
        plt.title("PS overlap")
    
    def positivity_check(self, bounds: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
        """Check overlap/positivity based on propensity score thresholds.
        
        Parameters
        ----------
        bounds : Tuple[float, float], default (0.05, 0.95)
            Lower and upper thresholds for positivity check
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with:
            - bounds: (low, high) thresholds used
            - share_below: fraction with PS < low
            - share_above: fraction with PS > high
            - flag: heuristic boolean True if the tails collectively exceed ~2%
        """
        low, high = bounds
        ps = self.propensity_scores
        share_low = float((ps < low).mean())
        share_high = float((ps > high).mean())
        flag = (share_low + share_high) > 0.02  # heuristic
        return {"bounds": bounds, "share_below": share_low, "share_above": share_high, "flag": bool(flag)}


class OutcomeModel:
    """A model for outcome prediction and related diagnostics.
    
    This class encapsulates outcome predictions and provides methods for:
    - Computing RMSE and MAE regression metrics
    - Extracting SHAP values for outcome prediction
    
    The class is returned by CausalEDA.outcome_fit() and provides a cleaner
    interface for outcome model analysis.
    """
    
    def __init__(self, 
                 predicted_outcomes: np.ndarray,
                 actual_outcomes: np.ndarray,
                 fitted_model: Any,
                 feature_names: List[str],
                 X_for_shap: Optional[np.ndarray] = None,
                 cat_features_for_shap: Optional[List[int]] = None):
        """Initialize OutcomeModel with fitted model artifacts.
        
        Parameters
        ----------
        predicted_outcomes : np.ndarray
            Array of predicted outcome values
        actual_outcomes : np.ndarray
            Array of actual outcome values
        fitted_model : Any
            The fitted outcome prediction model
        feature_names : List[str]
            Names of features used in the model (confounders only)
        X_for_shap : Optional[np.ndarray]
            Preprocessed feature matrix for SHAP computation
        cat_features_for_shap : Optional[List[int]]
            Categorical feature indices for SHAP computation
        """
        self.predicted_outcomes = predicted_outcomes
        self.actual_outcomes = actual_outcomes
        self.fitted_model = fitted_model
        self.feature_names = feature_names
        self.X_for_shap = X_for_shap
        self.cat_features_for_shap = cat_features_for_shap
    
    @property
    def scores(self) -> Dict[str, float]:
        """Compute regression metrics (RMSE and MAE) for outcome predictions.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
        """
        mse = mean_squared_error(self.actual_outcomes, self.predicted_outcomes)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.actual_outcomes, self.predicted_outcomes)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae)
        }
    
    @property
    def shap(self) -> pd.DataFrame:
        """Return SHAP values from the fitted outcome prediction model.
        
        SHAP values show the directional contribution of each feature to 
        outcome prediction, where positive values increase the predicted 
        outcome and negative values decrease it.
        
        Returns
        -------
        pd.DataFrame
            For CatBoost models: DataFrame with columns 'feature' and 'shap_mean',
            where 'shap_mean' represents the mean SHAP value across all samples.
            
            For sklearn models: DataFrame with columns 'feature' and 'importance'
            (absolute coefficient values, for backward compatibility).
            
        Raises
        ------
        RuntimeError
            If the fitted model does not support SHAP values extraction.
        """
        # Extract SHAP values or feature importance based on model type
        if isinstance(self.fitted_model, CatBoostRegressor):
            # Use CatBoost's SHAP values for directional feature contributions
            try:
                # Import Pool for SHAP computation
                from catboost import Pool
                
                # Check if we have the required data for SHAP computation
                if self.X_for_shap is None:
                    raise RuntimeError("Preprocessed data for SHAP computation not available.")
                
                # Create Pool object for SHAP computation (numeric-only after preprocessing)
                shap_pool = Pool(data=self.X_for_shap)
                
                # Get SHAP values - returns array of shape (n_samples, n_features + 1) 
                # where the last column is the bias term
                shap_values = self.fitted_model.get_feature_importance(type='ShapValues', data=shap_pool)
                
                # Remove bias term (last column) and compute mean SHAP values across samples
                # This gives us the average directional contribution of each feature
                shap_values_no_bias = shap_values[:, :-1]  # Remove bias column
                importance_values = np.mean(shap_values_no_bias, axis=0)  # Mean across samples
                
                feature_names = self.feature_names
                column_name = 'shap_mean'  # Use different column name to indicate SHAP values
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract SHAP values from CatBoost model: {e}")
        
        elif hasattr(self.fitted_model, 'named_steps') and hasattr(self.fitted_model.named_steps.get('clf'), 'coef_'):
            # Handle sklearn pipeline with linear regression
            try:
                clf = self.fitted_model.named_steps['clf']
                # For linear regression, use absolute coefficients as importance
                importance_values = np.abs(clf.coef_)
                
                # Need to map back to original feature names through preprocessing
                # For simplicity, if we have preprocessed features, we'll use the original feature names
                # and aggregate importance for one-hot encoded categorical features
                prep = self.fitted_model.named_steps.get('prep')
                if hasattr(prep, 'get_feature_names_out'):
                    try:
                        # Try to get feature names from preprocessor
                        transformed_names = prep.get_feature_names_out()
                        feature_names = [str(name) for name in transformed_names]
                    except:
                        # Fallback to original feature names if transformation fails
                        feature_names = self.feature_names
                        if len(importance_values) != len(feature_names):
                            # If lengths don't match due to one-hot encoding, we can't map back easily
                            # Just use indices as feature names
                            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                else:
                    feature_names = self.feature_names
                
                column_name = 'importance'  # Keep backward compatibility for sklearn models
                    
            except Exception as e:
                raise RuntimeError(f"Failed to extract feature importance from sklearn model: {e}")
        
        else:
            raise RuntimeError(f"Feature importance extraction not supported for model type: {type(self.fitted_model)}")
        
        # Ensure we have matching lengths
        if len(importance_values) != len(feature_names):
            # This can happen with preprocessing transformations
            # Create generic feature names if needed
            if len(importance_values) > len(feature_names):
                feature_names = feature_names + [f"transformed_feature_{i}" for i in range(len(feature_names), len(importance_values))]
            else:
                feature_names = feature_names[:len(importance_values)]
        
        # Create DataFrame with appropriate column name
        result_df = pd.DataFrame({
            'feature': feature_names,
            column_name: importance_values
        })
        
        # Sort appropriately: by absolute value for SHAP values (to show most impactful features),
        # by value for regular importance (higher is better)
        if column_name == 'shap_mean':
            # For SHAP values, sort by absolute value to show most impactful features first
            result_df = result_df.reindex(result_df[column_name].abs().sort_values(ascending=False).index)
        else:
            # For regular importance, sort by value (descending)
            result_df = result_df.sort_values(column_name, ascending=False)
        
        result_df = result_df.reset_index(drop=True)
        return result_df




# Optional lightweight dataclass for standalone usage, but CausalEDA also
# supports the existing causalkit.data.CausalData which uses `confounders`.
@dataclass
class CausalDataLite:
    """A minimal container for dataset roles used by CausalEDA.

    Attributes
    - df: The full pandas DataFrame containing treatment, outcome and covariates.
    - treatment: Column name of the binary treatment indicator (0/1).
    - target: Column name of the outcome variable.
    - confounders: List of covariate column names used to model treatment.
    """
    df: pd.DataFrame
    treatment: str
    target: str
    confounders: List[str]


def _extract_roles(data_obj: Any) -> Dict[str, Any]:
    """Extract dataset roles from various supported data containers.

    Accepts:
    - CausalDataLite
    - Project's CausalData (duck-typed: attributes df, treatment, outcome,
      and either confounders or confounders)
    - Any object exposing the same attributes/properties

    Returns a dict with keys: df, treatment, outcome, confounders.
    If both confounders/confounders are absent, it assumes all columns except
    treatment/outcome are confounders.
    """
    # Direct dataclass-like attributes
    df = getattr(data_obj, "df")
    treatment_attr = getattr(data_obj, "treatment")
    # Support both 'outcome' (project's CausalData) and 'target' (CausalDataLite)
    target_attr = getattr(data_obj, "outcome", None)
    if target_attr is None:
        target_attr = getattr(data_obj, "target")
    # If these are Series (as in causalkit.data.CausalData properties), convert to column names
    if isinstance(treatment_attr, pd.Series):
        treatment = treatment_attr.name
    else:
        treatment = treatment_attr
    if isinstance(target_attr, pd.Series):
        target = target_attr.name
    else:
        target = target_attr

    if hasattr(data_obj, "confounders") and getattr(data_obj, "confounders") is not None:
        confs = list(getattr(data_obj, "confounders"))
    elif hasattr(data_obj, "confounders") and getattr(data_obj, "confounders") is not None:
        # causalkit.data.CausalData.confounders returns a DataFrame or None; if it's a
        # DataFrame, use its columns; if it's a list/iterable, cast to list.
        cofs = getattr(data_obj, "confounders")
        if isinstance(cofs, pd.DataFrame):
            confs = list(cofs.columns)
        else:
            confs = list(cofs) if cofs is not None else []
    else:
        # Last resort: assume all columns except treatment/outcome are confounders
        confs = [c for c in df.columns if c not in {treatment, target}]

    return {"df": df, "treatment": treatment, "outcome": target, "confounders": confs}


class CausalEDA:
    """Exploratory diagnostics for causal designs with binary treatment.

    The class exposes methods to:
    
    - Summarize outcome by treatment and naive mean difference.
    - Estimate cross-validated propensity scores and assess treatment
      predictability (AUC) and positivity/overlap.
    - Inspect covariate balance via standardized mean differences (SMD)
      before/after IPTW weighting; visualize with a love plot.
    - Inspect weight distributions and effective sample size (ESS).
    """
    def __init__(self, data: Any, ps_model: Optional[Any] = None, n_splits: int = 5, random_state: int = 42):
        roles = _extract_roles(data)
        self.d = CausalDataLite(df=roles["df"], treatment=roles["treatment"], target=roles["outcome"], confounders=roles["confounders"])
        self.n_splits = n_splits
        self.random_state = random_state
        self.ps_model = ps_model or CatBoostClassifier(
            thread_count=-1,  # Use all available threads
            random_seed=random_state,
            verbose=False  # Suppress training output
        )

        # Preprocessing: always make features numeric via OneHotEncoder for categoricals
        X = self.d.df[self.d.confounders]
        num = X.select_dtypes(include=[np.number]).columns.tolist()
        cat = [c for c in X.columns if c not in num]

        num_transformer = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])
        self.preproc = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), cat),
            ],
            remainder="drop",
        )
        
        self.ps_pipe = Pipeline([("prep", self.preproc), ("clf", self.ps_model)])
        # Optional: clarify no native categorical indices are used post-OHE
        self.cat_features = None

    # ---------- basics ----------

    def data_shape(self) -> Dict[str, int]:
        """Return the shape information of the causal dataset.

        Returns a dict with:
        - n_rows: number of rows (observations) in the dataset
        - n_columns: number of columns (features) in the dataset
        
        This provides a quick overview of the dataset dimensions for 
        exploratory analysis and reporting purposes.
        
        Returns
        -------
        Dict[str, int]
            Dictionary containing 'n_rows' and 'n_columns' keys with 
            corresponding integer values representing the dataset dimensions.
        
        Examples
        --------
        >>> eda = CausalEDA(causal_data)
        >>> shape_info = eda.data_shape()
        >>> print(f"Dataset has {shape_info['n_rows']} rows and {shape_info['n_columns']} columns")
        """
        df = self.d.df
        n_rows, n_columns = df.shape
        return {"n_rows": n_rows, "n_columns": n_columns}


    def outcome_stats(self) -> pd.DataFrame:
        """Comprehensive outcome statistics grouped by treatment.

        Returns a DataFrame with detailed outcome statistics for each treatment group,
        including count, mean, std, min, various percentiles, and max.
        This method provides comprehensive outcome analysis and returns
        data in a clean DataFrame format suitable for reporting.

        Returns
        -------
        pd.DataFrame
            DataFrame with treatment groups as index and the following columns:
            - count: number of observations in each group
            - mean: average outcome value
            - std: standard deviation of outcome
            - min: minimum outcome value
            - p10: 10th percentile
            - p25: 25th percentile (Q1)
            - median: 50th percentile (median)
            - p75: 75th percentile (Q3)
            - p90: 90th percentile
            - max: maximum outcome value

        Examples
        --------
        >>> eda = CausalEDA(causal_data)
        >>> stats = eda.outcome_stats()
        >>> print(stats)
                count      mean       std       min       p10       p25    median       p75       p90       max
        treatment                                                                                                
        0        3000  5.123456  2.345678  0.123456  2.345678  3.456789  5.123456  6.789012  7.890123  9.876543
        1        2000  6.789012  2.456789  0.234567  3.456789  4.567890  6.789012  8.901234  9.012345  10.987654
        """
        df, t, y = self.d.df, self.d.treatment, self.d.target
        
        # Ensure treatment is numeric for grouping
        if not pd.api.types.is_numeric_dtype(df[t]):
            raise ValueError("Treatment must be numeric 0/1 for outcome_stats().")
        
        # Create grouped object for multiple operations
        grouped = df.groupby(t)[y]
        
        # Calculate basic statistics using built-in methods
        basic_stats = grouped.agg(['count', 'mean', 'std', 'min', 'median', 'max'])
        
        # Calculate percentiles separately to avoid pandas aggregation mixing issues
        p10 = grouped.quantile(0.10)
        p25 = grouped.quantile(0.25) 
        p75 = grouped.quantile(0.75)
        p90 = grouped.quantile(0.90)
        
        # Combine all statistics into a single DataFrame
        stats_df = pd.DataFrame({
            'count': basic_stats['count'],
            'mean': basic_stats['mean'],
            'std': basic_stats['std'],
            'min': basic_stats['min'],
            'p10': p10,
            'p25': p25,
            'median': basic_stats['median'],
            'p75': p75,
            'p90': p90,
            'max': basic_stats['max']
        })
        
        # Ensure the index is named appropriately
        stats_df.index.name = 'treatment'
        
        return stats_df

    # ---------- propensity & overlap ----------
    def fit_propensity(self) -> 'PropensityModel':
        """Estimate cross-validated propensity scores P(T=1|X).

        Uses a preprocessing+CatBoost classifier pipeline with stratified K-fold
        cross_val_predict to generate out-of-fold probabilities. CatBoost uses
        all available threads and handles categorical features natively. Returns a
        PropensityModel instance containing propensity scores and diagnostic methods.
        
        Returns
        -------
        PropensityModel
            A PropensityModel instance with methods for:
            - roc_auc: ROC AUC score property
            - shap: SHAP values DataFrame property  
            - ps_graph(): method to plot propensity score overlap
            - positivity_check(): method to check positivity/overlap
        """
        df = self.d.df
        X = df[self.d.confounders]
        t = df[self.d.treatment].astype(int).values
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Special handling for CatBoost to properly pass categorical features
        if isinstance(self.ps_model, CatBoostClassifier):
            # For CatBoost, we need custom cross-validation to pass cat_features properly
            import warnings
            
            ps = np.zeros(len(X))
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    
                    for train_idx, test_idx in cv.split(X, t):
                        # Prepare data for this fold
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        t_train = t[train_idx]
                        
                        # Apply preprocessing
                        X_train_prep = self.preproc.fit_transform(X_train)
                        X_test_prep = self.preproc.transform(X_test)
                        
                        # Create and train CatBoost model for this fold
                        model = CatBoostClassifier(
                            thread_count=-1,
                            random_seed=self.random_state,
                            verbose=False
                        )
                        
                        # All features are numeric after preprocessing; no cat_features needed
                        model.fit(X_train_prep, t_train)
                        ps[test_idx] = model.predict_proba(X_test_prep)[:, 1]
        else:
            # Use standard sklearn pipeline for non-CatBoost models
            import warnings
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    ps = cross_val_predict(self.ps_pipe, X, t, cv=cv, method="predict_proba")[:, 1]
        
        # clip away from 0/1 for stability
        ps = np.clip(ps, 1e-6, 1 - 1e-6)
        self._ps = ps
        
        # Train a final model on the full dataset for feature importance
        # This provides consistent feature importance across the entire dataset
        if isinstance(self.ps_model, CatBoostClassifier):
            # Apply preprocessing to full dataset
            X_full_prep = self.preproc.fit_transform(X)
            
            # Create and train final model
            final_model = CatBoostClassifier(
                thread_count=-1,
                random_seed=self.random_state,
                verbose=False
            )
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    final_model.fit(X_full_prep, t)
            
            # Store the trained model and data needed for SHAP computation
            self._fitted_model = final_model
            self._feature_names = X.columns.tolist()
            self._X_for_shap = X_full_prep  # Store preprocessed data for SHAP
        else:
            # For non-CatBoost models, fit the pipeline on full data
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    self.ps_pipe.fit(X, t)
            self._fitted_model = self.ps_pipe
            self._feature_names = X.columns.tolist()
        
        # Create and return PropensityModel instance
        return PropensityModel(
            propensity_scores=ps,
            treatment_values=t,
            fitted_model=self._fitted_model,
            feature_names=self._feature_names,
            X_for_shap=getattr(self, '_X_for_shap', None)
        )

    def outcome_fit(self, outcome_model: Optional[Any] = None) -> 'OutcomeModel':
        """Fit a regression model to predict outcome from confounders only.

        Uses a preprocessing+CatBoost regressor pipeline with K-fold
        cross_val_predict to generate out-of-fold predictions. CatBoost uses
        all available threads and handles categorical features natively. Returns an
        OutcomeModel instance containing predicted outcomes and diagnostic methods.
        
        The outcome model predicts the baseline outcome from confounders only,
        excluding treatment. This is essential for proper causal analysis.
        
        Parameters
        ----------
        outcome_model : Optional[Any]
            Custom regression model to use. If None, uses CatBoostRegressor.
            
        Returns
        -------
        OutcomeModel
            An OutcomeModel instance with methods for:
            - scores: RMSE and MAE regression metrics
            - shap: SHAP values DataFrame property for outcome prediction
        """
        df = self.d.df
        # Features: confounders only (treatment excluded for proper causal analysis)
        X = df[self.d.confounders]
        y = df[self.d.target].values
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Default to CatBoostRegressor if no custom model provided
        if outcome_model is None:
            outcome_model = CatBoostRegressor(
                thread_count=-1,
                random_seed=self.random_state,
                verbose=False
            )
        
        # Identify numeric and categorical features for preprocessing
        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = [c for c in X.columns if c not in num_features]
        
        # Setup preprocessing: always OHE categoricals so model input is numeric
        num_transformer = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])
        preproc = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), cat_features),
            ],
            remainder="drop",
        )
        
        # Special handling for CatBoost to properly pass categorical features
        if isinstance(outcome_model, CatBoostRegressor):
            import warnings
            
            predictions = np.zeros(len(X))
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    
                    for train_idx, test_idx in cv.split(X, y):
                        # Prepare data for this fold
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train = y[train_idx]
                        
                        # Apply preprocessing
                        X_train_prep = preproc.fit_transform(X_train)
                        X_test_prep = preproc.transform(X_test)
                        
                        # Create and train CatBoost model for this fold
                        model = CatBoostRegressor(
                            thread_count=-1,
                            random_seed=self.random_state,
                            verbose=False
                        )
                        
                        # All features are numeric after preprocessing; no cat_features needed
                        model.fit(X_train_prep, y_train)
                        predictions[test_idx] = model.predict(X_test_prep)
        else:
            # Use standard sklearn pipeline for non-CatBoost models
            pipeline = Pipeline([("prep", preproc), ("reg", outcome_model)])
            import warnings
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    predictions = cross_val_predict(pipeline, X, y, cv=cv)
        
        self._outcome_predictions = predictions
        
        # Train a final model on the full dataset for SHAP computation
        if isinstance(outcome_model, CatBoostRegressor):
            # Apply preprocessing to full dataset
            X_full_prep = preproc.fit_transform(X)
            
            # Create and train final model
            final_model = CatBoostRegressor(
                thread_count=-1,
                random_seed=self.random_state,
                verbose=False
            )
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    final_model.fit(X_full_prep, y)
            
            # Store the trained model and data needed for SHAP computation
            self._outcome_fitted_model = final_model
            self._outcome_feature_names = X.columns.tolist()
            self._outcome_X_for_shap = X_full_prep  # Store preprocessed data for SHAP
        else:
            # For non-CatBoost models, fit the pipeline on full data
            pipeline = Pipeline([("prep", preproc), ("reg", outcome_model)])
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    pipeline.fit(X, y)
            self._outcome_fitted_model = pipeline
            self._outcome_feature_names = X.columns.tolist()
        
        # Create and return OutcomeModel instance
        return OutcomeModel(
            predicted_outcomes=predictions,
            actual_outcomes=y,
            fitted_model=self._outcome_fitted_model,
            feature_names=self._outcome_feature_names,
            X_for_shap=getattr(self, '_outcome_X_for_shap', None)
        )

    def confounders_roc_auc(self, ps: Optional[np.ndarray] = None) -> float:
        """Compute ROC AUC of treatment assignment vs. estimated propensity score.

        Interpretation: Higher AUC means treatment is more predictable from confounders,
        indicating stronger systematic differences between groups (potential
        confounding). Values near 0.5 suggest random-like assignment.
        """
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps_model = self.fit_propensity()
                ps = ps_model.propensity_scores
        # AUC against actual treatment
        t = self.d.df[self.d.treatment].astype(int).values
        return float(roc_auc_score(t, ps))

    def positivity_check(self, ps: Optional[np.ndarray] = None, bounds: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
        """Check overlap/positivity based on propensity score thresholds.

        Returns a dict with:
        - bounds: (low, high) thresholds used
        - share_below: fraction with PS < low
        - share_above: fraction with PS > high
        - flag: heuristic boolean True if the tails collectively exceed ~2%
        """
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps_model = self.fit_propensity()
                ps = ps_model.propensity_scores
        low, high = bounds
        share_low = float((ps < low).mean())
        share_high = float((ps > high).mean())
        flag = (share_low + share_high) > 0.02  # heuristic
        return {"bounds": bounds, "share_below": share_low, "share_above": share_high, "flag": bool(flag)}

    def plot_ps_overlap(self, ps: Optional[np.ndarray] = None):
        """Plot overlaid histograms of propensity scores for treated vs control.

        Useful to visually assess group overlap. Does not return data; it draws
        on the current matplotlib figure.
        """
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps_model = self.fit_propensity()
                ps = ps_model.propensity_scores
        df = self.d.df
        t = df[self.d.treatment].astype(int).values
        plt.figure()
        plt.hist(ps[t == 1], bins=30, alpha=0.5, density=True, label="treated")
        plt.hist(ps[t == 0], bins=30, alpha=0.5, density=True, label="control")
        plt.xlabel("Propensity score")
        plt.ylabel("Density")
        plt.legend()
        plt.title("PS overlap")

    # ---------- balance ----------

    def confounders_means(self) -> pd.DataFrame:
        """Comprehensive confounders balance assessment with means by treatment group.

        Returns a DataFrame with detailed balance information including:
        - Mean values of each confounder for control group (treatment=0)
        - Mean values of each confounder for treated group (treatment=1)
        - Absolute difference between treatment groups
        - Standardized Mean Difference (SMD) for formal balance assessment
        
        This method provides a comprehensive view of confounder balance by showing
        the actual mean values alongside the standardized differences, making it easier
        to understand both the magnitude and direction of imbalances.

        Returns
        -------
        pd.DataFrame
            DataFrame with confounders as index and the following columns:
            - mean_t_0: mean value for control group (treatment=0)
            - mean_t_1: mean value for treated group (treatment=1)  
            - abs_diff: absolute difference abs(mean_t_1 - mean_t_0)
            - smd: standardized mean difference (Cohen's d)
            
        Notes
        -----
        SMD values > 0.1 in absolute value typically indicate meaningful imbalance.
        Categorical variables are automatically converted to dummy variables.
        
        Examples
        --------
        >>> eda = CausalEDA(causal_data)
        >>> balance = eda.confounders_means()
        >>> print(balance.head())
                     mean_t_0  mean_t_1  abs_diff       smd
        confounders                                       
        age              29.5      31.2      1.7     0.085
        income        45000.0   47500.0   2500.0     0.125
        education         0.25      0.35      0.1     0.215
        """
        df = self.d.df
        X = df[self.d.confounders]
        t = df[self.d.treatment].astype(int).values
        
        # Convert categorical variables to dummy variables for analysis
        X_num = pd.get_dummies(X, drop_first=False)
        
        rows = []
        for col in X_num.columns:
            x = X_num[col].values.astype(float)
            
            # Calculate means for each treatment group
            mean_t_0 = x[t == 0].mean()
            mean_t_1 = x[t == 1].mean()
            
            # Calculate absolute difference
            abs_diff = abs(mean_t_1 - mean_t_0)
            
            # Calculate standardized mean difference (SMD)
            v_control = x[t == 0].var(ddof=1) if len(x[t == 0]) > 1 else 0.0
            v_treated = x[t == 1].var(ddof=1) if len(x[t == 1]) > 1 else 0.0
            pooled_std = np.sqrt((v_control + v_treated) / 2)
            smd = (mean_t_1 - mean_t_0) / pooled_std if pooled_std > 0 else 0.0
            
            rows.append({
                "confounders": col,
                "mean_t_0": mean_t_0,
                "mean_t_1": mean_t_1, 
                "abs_diff": abs_diff,
                "smd": smd
            })
        
        # Create DataFrame and set confounders as index
        balance_df = pd.DataFrame(rows)
        balance_df = balance_df.set_index("confounders")
        
        # Sort by absolute SMD value (most imbalanced first)
        balance_df = balance_df.reindex(
            balance_df['smd'].abs().sort_values(ascending=False).index
        )
        
        return balance_df




    def outcome_plots(self,
                      treatment: Optional[str] = None,
                      target: Optional[str] = None,
                      bins: int = 30,
                      density: bool = True,
                      alpha: float = 0.5,
                      figsize: Tuple[float, float] = (7, 4),
                      sharex: bool = True) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot the distribution of the outcome for every treatment on one plot,
        and also produce a boxplot by treatment to visualize outliers.

        Parameters
        ----------
        treatment : Optional[str]
            Treatment column name. Defaults to the treatment stored in the CausalEDA data.
        target : Optional[str]
            Target/outcome column name. Defaults to the outcome stored in the CausalEDA data.
        bins : int
            Number of bins for histograms when the outcome is numeric.
        density : bool
            Whether to normalize histograms to form a density.
        alpha : float
            Transparency for overlaid histograms.
        figsize : tuple
            Figure size for the plots.
        sharex : bool
            If True and the outcome is numeric, use the same x-limits across treatments.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
            (fig_distribution, fig_boxplot)
        """
        df = self.d.df
        t_col = treatment or self.d.treatment
        y_col = target or self.d.target

        if t_col not in df.columns or y_col not in df.columns:
            raise ValueError("Specified treatment/outcome columns not found in DataFrame.")

        # Determine unique treatments preserving natural sort
        treatments = pd.unique(df[t_col])

        # Distribution plot (overlayed)
        fig1 = plt.figure(figsize=figsize)
        ax1 = fig1.gca()

        # Only support numeric outcome for histogram/boxplot in this minimal implementation
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            # Fallback: draw normalized bars per treatment for categorical outcome
            # Compute frequency of outcome values per treatment and stack them side-by-side
            vals = pd.unique(df[y_col])
            vals_sorted = sorted(vals, key=lambda v: (str(type(v)), v))
            width = 0.8 / max(1, len(treatments))
            x = np.arange(len(vals_sorted))
            for i, tr in enumerate(treatments):
                sub = df[df[t_col] == tr][y_col]
                counts = pd.Series(sub).value_counts(normalize=True)
                heights = [counts.get(v, 0.0) for v in vals_sorted]
                ax1.bar(x + i * width, heights, width=width, alpha=alpha, label=str(tr))
            ax1.set_xticks(x + (len(treatments) - 1) * width / 2)
            ax1.set_xticklabels([str(v) for v in vals_sorted])
            ax1.set_ylabel("Proportion")
            ax1.set_xlabel(str(y_col))
            ax1.set_title("Target distribution by treatment (categorical)")
            ax1.legend(title=str(t_col))
        else:
            # Numeric outcome: overlay histograms/density
            # Determine common x-limits if sharex
            xmin, xmax = None, None
            if sharex:
                xmin = float(df[y_col].min())
                xmax = float(df[y_col].max())
            for tr in treatments:
                y_vals = df.loc[df[t_col] == tr, y_col].dropna().values
                if len(y_vals) == 0:
                    continue
                ax1.hist(y_vals, bins=bins, density=density, alpha=alpha, label=str(tr), range=(xmin, xmax) if sharex else None)
            ax1.set_xlabel(str(y_col))
            ax1.set_ylabel("Density" if density else "Count")
            ax1.set_title("Target distribution by treatment")
            ax1.legend(title=str(t_col))

        # Boxplot by treatment
        fig2 = plt.figure(figsize=figsize)
        ax2 = fig2.gca()
        # Create data in order of treatments
        data = [df.loc[df[t_col] == tr, y_col].dropna().values for tr in treatments]
        ax2.boxplot(data, labels=[str(tr) for tr in treatments], showfliers=True)
        ax2.set_xlabel(str(t_col))
        ax2.set_ylabel(str(y_col))
        ax2.set_title("Target by treatment (boxplot)")

        return fig1, fig2

    def treatment_features(self) -> pd.DataFrame:
        """Return SHAP values from the fitted propensity score model.
        
        This method extracts SHAP values from the propensity score model
        that was trained during fit_propensity(). SHAP values show the directional
        contribution of each feature to treatment assignment prediction, where
        positive values increase treatment probability and negative values decrease it.
        
        Returns
        -------
        pd.DataFrame
            For CatBoost models: DataFrame with columns 'feature' and 'shap_mean', 
            where 'shap_mean' represents the mean SHAP value across all samples.
            Positive values indicate features that increase treatment probability,
            negative values indicate features that decrease treatment probability.
            
            For sklearn models: DataFrame with columns 'feature' and 'importance'
            (absolute coefficient values, for backward compatibility).
            
        Raises
        ------
        RuntimeError
            If fit_propensity() has not been called yet, or if the fitted
            model does not support SHAP values extraction.
            
        Examples
        --------
        >>> eda = CausalEDA(data)
        >>> ps = eda.fit_propensity()  # Must be called first
        >>> shap_df = eda.treatment_features()
        >>> print(shap_df.head())
           feature  shap_mean
        0  age         0.45  # Positive: increases treatment prob
        1  income     -0.32  # Negative: decreases treatment prob
        2  education   0.12  # Positive: increases treatment prob
        """
        # Check if model has been fitted
        if not hasattr(self, '_fitted_model') or self._fitted_model is None:
            raise RuntimeError("No fitted propensity model found. Please call fit_propensity() first.")
        
        if not hasattr(self, '_feature_names') or self._feature_names is None:
            raise RuntimeError("Feature names not available. Please call fit_propensity() first.")
        
        # Extract SHAP values or feature importance based on model type
        if isinstance(self._fitted_model, CatBoostClassifier):
            # Use CatBoost's SHAP values for directional feature contributions
            try:
                # Import Pool for SHAP computation
                from catboost import Pool
                
                # Check if we have the required data for SHAP computation
                if not hasattr(self, '_X_for_shap') or self._X_for_shap is None:
                    raise RuntimeError("Preprocessed data for SHAP computation not available. Please call fit_propensity() first.")
                
                # Create Pool object for SHAP computation (numeric-only after preprocessing)
                shap_pool = Pool(data=self._X_for_shap)
                
                # Get SHAP values - returns array of shape (n_samples, n_features + 1) 
                # where the last column is the bias term
                shap_values = self._fitted_model.get_feature_importance(type='ShapValues', data=shap_pool)
                
                # Remove bias term (last column) and compute mean SHAP values across samples
                # This gives us the average directional contribution of each feature
                shap_values_no_bias = shap_values[:, :-1]  # Remove bias column
                importance_values = np.mean(shap_values_no_bias, axis=0)  # Mean across samples
                
                feature_names = self._feature_names
                column_name = 'shap_mean'  # Use different column name to indicate SHAP values
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract SHAP values from CatBoost model: {e}")
        
        elif hasattr(self._fitted_model, 'named_steps') and hasattr(self._fitted_model.named_steps.get('clf'), 'coef_'):
            # Handle sklearn pipeline with logistic regression
            try:
                clf = self._fitted_model.named_steps['clf']
                # For logistic regression, use absolute coefficients as importance
                importance_values = np.abs(clf.coef_[0])
                
                # Need to map back to original feature names through preprocessing
                # For simplicity, if we have preprocessed features, we'll use the original feature names
                # and aggregate importance for one-hot encoded categorical features
                prep = self._fitted_model.named_steps.get('prep')
                if hasattr(prep, 'get_feature_names_out'):
                    try:
                        # Try to get feature names from preprocessor
                        transformed_names = prep.get_feature_names_out()
                        feature_names = [str(name) for name in transformed_names]
                    except:
                        # Fallback to original feature names if transformation fails
                        feature_names = self._feature_names
                        if len(importance_values) != len(feature_names):
                            # If lengths don't match due to one-hot encoding, we can't map back easily
                            # Just use indices as feature names
                            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                else:
                    feature_names = self._feature_names
                
                column_name = 'importance'  # Keep backward compatibility for sklearn models
                    
            except Exception as e:
                raise RuntimeError(f"Failed to extract feature importance from sklearn model: {e}")
        
        else:
            raise RuntimeError(f"Feature importance extraction not supported for model type: {type(self._fitted_model)}")
        
        # Ensure we have matching lengths
        if len(importance_values) != len(feature_names):
            # This can happen with preprocessing transformations
            # Create generic feature names if needed
            if len(importance_values) > len(feature_names):
                feature_names = feature_names + [f"transformed_feature_{i}" for i in range(len(feature_names), len(importance_values))]
            else:
                feature_names = feature_names[:len(importance_values)]
        
        # Create DataFrame with appropriate column name
        result_df = pd.DataFrame({
            'feature': feature_names,
            column_name: importance_values
        })
        
        # Sort appropriately: by absolute value for SHAP values (to show most impactful features),
        # by value for regular importance (higher is better)
        if column_name == 'shap_mean':
            # For SHAP values, sort by absolute value to show most impactful features first
            result_df = result_df.reindex(result_df[column_name].abs().sort_values(ascending=False).index)
        else:
            # For regular importance, sort by value (descending)
            result_df = result_df.sort_values(column_name, ascending=False)
        
        result_df = result_df.reset_index(drop=True)
        return result_df

