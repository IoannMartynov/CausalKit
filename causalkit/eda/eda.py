"""EDA utilities for causal analysis (propensity, overlap, balance, weights).

This module provides a lightweight CausalEDA class to quickly assess whether a
binary treatment problem is suitable for causal effect estimation. The outputs
focus on interpretability: treatment predictability, overlap/positivity,
covariate balance before/after weighting, and basic data health.

What the main outputs mean
- summaries(): Dict with treatment_rate (share of treated),
  outcome_by_treatment (count/mean/std within treatment groups), and
  naive_diff (treated mean minus control mean; a biased estimate if confounding exists).
- fit_propensity(): Numpy array of cross-validated propensity scores P(T=1|X).
- treatment_predictability_auc(): Float AUC of treatment vs. propensity score.
  Higher AUC implies treatment is predictable from X (more confounding risk).
- positivity_check(): Dict with bounds, share_below, share_above, and flag.
  It reports what share of units have PS outside [low, high]; a large share
  signals poor overlap (violated positivity).
- plot_ps_overlap(): Overlaid histograms of PS for treated vs control.
- balance_table(): DataFrame with standardized mean differences (SMD) for each
  covariate and boolean flags if |SMD|>0.1 indicating imbalance.
- design_report(): One-shot dict aggregating all of the above, convenient for
  quick inspection and logging.

Note: The class accepts either the project’s CausalData object (duck-typed) or a
CausalDataLite with explicit fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


class DesignReport(dict):
    """A dictionary-like container for design report data with a summary() method.
    
    This class extends the standard dict to provide a summary() method that formats
    the design report data into beautiful, readable text while maintaining full
    backward compatibility with dictionary operations.
    """
    
    def summary(self) -> str:
        """Generate a beautiful text summary of the design report.
        
        Returns
        -------
        str
            A formatted text summary of all diagnostic information in the report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CAUSAL DESIGN REPORT SUMMARY")
        lines.append("=" * 60)
        
        # 1. Treatment and Outcome Summary
        if 'summaries' in self:
            summ = self['summaries']
            lines.append("\n📊 TREATMENT & OUTCOME SUMMARY")
            lines.append("-" * 30)
            
            treatment_rate = summ.get('treatment_rate', 'N/A')
            if isinstance(treatment_rate, (int, float)):
                lines.append(f"Treatment Rate: {treatment_rate:.1%}")
            else:
                lines.append(f"Treatment Rate: {treatment_rate}")
            
            naive_diff = summ.get('naive_diff', 'N/A')
            if isinstance(naive_diff, (int, float)) and not pd.isna(naive_diff):
                lines.append(f"Naive Difference (Treated - Control): {naive_diff:.4f}")
            else:
                lines.append(f"Naive Difference: {naive_diff}")
            
            # Outcome by treatment table
            if 'outcome_by_treatment' in summ:
                outcome_table = summ['outcome_by_treatment']
                if isinstance(outcome_table, pd.DataFrame) and not outcome_table.empty:
                    lines.append("\nOutcome by Treatment:")
                    lines.append(str(outcome_table.round(4)))
        
        # 2. Treatment Predictability
        if 'treat_auc' in self:
            auc = self['treat_auc']
            lines.append(f"\n🎯 TREATMENT PREDICTABILITY")
            lines.append("-" * 30)
            if isinstance(auc, (int, float)):
                lines.append(f"Treatment AUC: {auc:.4f}")
                if auc > 0.8:
                    lines.append("  ⚠️  High predictability - strong confounding risk")
                elif auc > 0.65:
                    lines.append("  ⚡ Moderate predictability - some confounding risk")
                else:
                    lines.append("  ✅ Low predictability - minimal confounding risk")
            else:
                lines.append(f"Treatment AUC: {auc}")
        
        # 3. Positivity/Overlap Assessment
        if 'positivity' in self:
            pos = self['positivity']
            lines.append(f"\n🔄 POSITIVITY/OVERLAP ASSESSMENT")
            lines.append("-" * 30)
            
            bounds = pos.get('bounds', 'N/A')
            share_below = pos.get('share_below', 'N/A')
            share_above = pos.get('share_above', 'N/A')
            flag = pos.get('flag', False)
            
            lines.append(f"Propensity Score Bounds: {bounds}")
            if isinstance(share_below, (int, float)) and isinstance(share_above, (int, float)):
                lines.append(f"Share Below Lower Bound: {share_below:.1%}")
                lines.append(f"Share Above Upper Bound: {share_above:.1%}")
                total_extreme = share_below + share_above
                lines.append(f"Total in Extreme Regions: {total_extreme:.1%}")
                
                if flag:
                    lines.append("  ⚠️  Poor overlap detected - positivity violation risk")
                else:
                    lines.append("  ✅ Good overlap - positivity assumption satisfied")
            else:
                lines.append(f"Share Below: {share_below}, Share Above: {share_above}")
        
        # 4. Balance Assessment
        if 'balance' in self:
            balance = self['balance']
            lines.append(f"\n⚖️  COVARIATE BALANCE ASSESSMENT")
            lines.append("-" * 30)
            
            if isinstance(balance, pd.DataFrame) and not balance.empty:
                # Summary statistics - calculate imbalanced variables from SMD values
                imbalanced_count = (balance['SMD'].abs() > 0.1).sum() if 'SMD' in balance.columns else 0
                total_vars = len(balance)
                
                lines.append(f"Total Variables: {total_vars}")
                lines.append(f"Imbalanced Variables: {imbalanced_count}")
                
                if imbalanced_count == 0:
                    lines.append("  ✅ Perfect balance - all variables well balanced")
                elif imbalanced_count <= total_vars * 0.2:
                    lines.append("  ⚡ Good balance - few imbalanced variables")
                else:
                    lines.append("  ⚠️  Poor balance - many imbalanced variables")
                
                # Show worst balanced variables
                balance_abs = balance.copy()
                balance_abs['SMD_abs'] = balance['SMD'].abs()
                worst_imbalanced = balance_abs.nlargest(3, 'SMD_abs', keep='first')
                if not worst_imbalanced.empty:
                    lines.append("\nMost Imbalanced Variables:")
                    for _, row in worst_imbalanced.iterrows():
                        smd = row.get('SMD', 'N/A')
                        var_name = row.get('covariate', 'Unknown')
                        if isinstance(smd, (int, float)):
                            lines.append(f"  {var_name}: {smd:.3f}")
                        else:
                            lines.append(f"  {var_name}: {smd}")
        
        # Summary and recommendations
        lines.append(f"\n📋 OVERALL ASSESSMENT")
        lines.append("-" * 30)
        
        # Simple scoring system for overall quality
        issues = []
        if 'treat_auc' in self and isinstance(self['treat_auc'], (int, float)) and self['treat_auc'] > 0.8:
            issues.append("High treatment predictability")
        if 'positivity' in self and self['positivity'].get('flag', False):
            issues.append("Poor overlap/positivity")
        if 'balance' in self and isinstance(self['balance'], pd.DataFrame):
            balance_df = self['balance']
            imbalanced_count = (balance_df['SMD'].abs() > 0.1).sum() if 'SMD' in balance_df.columns else 0
            if imbalanced_count > len(balance_df) * 0.2:  # More than 20% still imbalanced
                issues.append("Poor covariate balance")
        
        if not issues:
            lines.append("✅ Design looks good for causal inference!")
            lines.append("   All key assumptions appear to be satisfied.")
        else:
            lines.append("⚠️  Design has potential issues:")
            for issue in issues:
                lines.append(f"   - {issue}")
            lines.append("   Consider additional robustness checks or sensitivity analysis.")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Optional lightweight dataclass for standalone usage, but CausalEDA also
# supports the existing causalkit.data.CausalData which uses `cofounders`.
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
      and either confounders or cofounders)
    - Any object exposing the same attributes/properties

    Returns a dict with keys: df, treatment, outcome, confounders.
    If both confounders/cofounders are absent, it assumes all columns except
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
    elif hasattr(data_obj, "cofounders") and getattr(data_obj, "cofounders") is not None:
        # causalkit.data.CausalData.cofounders returns a DataFrame or None; if it's a
        # DataFrame, use its columns; if it's a list/iterable, cast to list.
        cofs = getattr(data_obj, "cofounders")
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
    - Produce a one-shot design_report() that aggregates the above.
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

        # Preprocessing for CatBoost - identify categorical features for native handling
        X = self.d.df[self.d.confounders]
        num = X.select_dtypes(include=[np.number]).columns.tolist()
        cat = [c for c in X.columns if c not in num]
        self.cat_features = [X.columns.get_loc(c) for c in cat] if cat else None
        
        # For CatBoost, we can use minimal preprocessing since it handles categoricals natively
        if isinstance(self.ps_model, CatBoostClassifier):
            # Only scale numeric features, keep categoricals as-is for CatBoost
            if num:
                num_transformer = Pipeline(steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True))
                ])
                self.preproc = ColumnTransformer(
                    transformers=[
                        ("num", num_transformer, num),
                        ("cat", "passthrough", cat),
                    ],
                    remainder="drop",
                )
            else:
                # All categorical, no preprocessing needed
                self.preproc = "passthrough"
        else:
            # Fallback preprocessing for other models (like LogisticRegression)
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

    # ---------- basics ----------


    def summaries(self) -> Dict[str, Any]:
        """Outcome summaries by treatment.

        Returns a dict with:
        - treatment_rate: share of treated units (mean of 0/1 treatment)
        - outcome_by_treatment: pandas DataFrame with count/mean/std of outcome per treatment group
        - naive_diff: treated mean minus control mean (unadjusted, potentially biased)
        """
        df, t, y = self.d.df, self.d.treatment, self.d.target
        if not pd.api.types.is_numeric_dtype(df[t]):
            raise ValueError("Treatment must be numeric 0/1 for summaries().")
        treat_rate = df[t].mean()
        grp = df.groupby(t)[y].agg(["count", "mean", "std"])
        # If either group missing, fill with nan-safe
        naive_diff = np.nan
        try:
            naive_diff = grp.loc[1, "mean"] - grp.loc[0, "mean"]
        except Exception:
            pass
        return {"treatment_rate": treat_rate, "outcome_by_treatment": grp, "naive_diff": naive_diff}

    # ---------- propensity & overlap ----------
    def fit_propensity(self) -> np.ndarray:
        """Estimate cross-validated propensity scores P(T=1|X).

        Uses a preprocessing+CatBoost classifier pipeline with stratified K-fold
        cross_val_predict to generate out-of-fold probabilities. CatBoost uses
        all available threads and handles categorical features natively. Returns a
        numpy array of shape (n_rows,) with values clipped to (1e-6, 1-1e-6).
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
                        
                        # Identify categorical features after preprocessing
                        if self.cat_features is not None:
                            # Map original categorical feature indices to preprocessed data
                            num_features = X.select_dtypes(include=[np.number]).shape[1]
                            cat_features_prep = list(range(num_features, X_train_prep.shape[1]))
                        else:
                            cat_features_prep = None
                        
                        model.fit(X_train_prep, t_train, cat_features=cat_features_prep)
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
            
            # Identify categorical features after preprocessing
            if self.cat_features is not None:
                # Map original categorical feature indices to preprocessed data
                num_features = X.select_dtypes(include=[np.number]).shape[1]
                cat_features_prep = list(range(num_features, X_full_prep.shape[1]))
            else:
                cat_features_prep = None
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    final_model.fit(X_full_prep, t, cat_features=cat_features_prep)
            
            # Store the trained model and data needed for SHAP computation
            self._fitted_model = final_model
            self._feature_names = X.columns.tolist()
            self._X_for_shap = X_full_prep  # Store preprocessed data for SHAP
            self._cat_features_for_shap = cat_features_prep  # Store categorical features info
        else:
            # For non-CatBoost models, fit the pipeline on full data
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    self.ps_pipe.fit(X, t)
            self._fitted_model = self.ps_pipe
            self._feature_names = X.columns.tolist()
        
        return ps

    def treatment_predictability_auc(self, ps: Optional[np.ndarray] = None) -> float:
        """Compute AUC of treatment assignment vs. estimated propensity score.

        Interpretation: Higher AUC means treatment is more predictable from X,
        indicating stronger systematic differences between groups (potential
        confounding). Values near 0.5 suggest random-like assignment.
        """
        if ps is None:
            ps = getattr(self, "_ps", None)
            if ps is None:
                ps = self.fit_propensity()
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
                ps = self.fit_propensity()
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
                ps = self.fit_propensity()
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

    def balance_table(self) -> pd.DataFrame:
        """Compute standardized mean differences (SMD) for covariate balance assessment.

        Returns a DataFrame with columns:
        - covariate: (possibly one-hot) covariate name
        - SMD: standardized mean difference between treated and control groups
        
        Lower |SMD| indicates better balance between treated and control groups.
        SMD values > 0.1 in absolute value typically indicate meaningful imbalance.
        """
        df = self.d.df
        X = df[self.d.confounders]
        t = df[self.d.treatment].astype(int).values
        # convert cats to dummies for SMDs
        X_num = pd.get_dummies(X, drop_first=False)
        rows = []
        for col in X_num.columns:
            x = X_num[col].values.astype(float)
            # compute unweighted SMD
            m1 = x[t == 1].mean()
            v1 = x[t == 1].var(ddof=1)
            m0 = x[t == 0].mean()
            v0 = x[t == 0].var(ddof=1)
            smd = (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0
            rows.append({"covariate": col, "SMD": smd})
        tab = pd.DataFrame(rows).sort_values("SMD", key=lambda s: s.abs(), ascending=False)
        return tab



    def plot_target_by_treatment(self,
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
                
                # Create Pool object for SHAP computation
                shap_pool = Pool(data=self._X_for_shap, cat_features=self._cat_features_for_shap)
                
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

    # ---------- one-shot driver ----------
    def design_report(self) -> DesignReport:
        """Run a full set of diagnostics and return a consolidated report.

        Returns a DesignReport (dict-like) with keys:
        - summaries: output of summaries()
        - treat_auc: float from treatment_predictability_auc()
        - positivity: output of positivity_check()
        - balance: output of balance_table()
        
        The returned object supports .summary() for beautiful text formatting
        and behaves like a regular dictionary for backward compatibility.
        Useful for quick inspection and logging in notebooks or pipelines.
        
        Example:
            report = eda.design_report()
            print(report.summary())  # Beautiful formatted text
            print(report['treat_auc'])  # Access like a dictionary
        """
        ps = self.fit_propensity()
        return DesignReport({
            "summaries": self.summaries(),
            "treat_auc": self.treatment_predictability_auc(ps),
            "positivity": self.positivity_check(ps),
            "balance": self.balance_table(),
        })
