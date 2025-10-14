# EDA Module

The `causalis.eda` module provides exploratory diagnostics for causal designs with binary treatment. It helps assess treatment predictability, overlap/positivity, covariate balance, and outcome modeling quality before running inference.

## Overview

Key components:

- `CausalEDA`: High-level interface for EDA on CausalData or a lightweight container
- `CausalDataLite`: Minimal data container compatible with CausalEDA

Main capabilities:
- Outcome group statistics by treatment
- Cross-validated propensity scores with ROC AUC and positivity checks
- Covariate balance diagnostics (means, absolute diffs, and standardized mean differences)
- Outcome model fit diagnostics (RMSE, MAE) and SHAP-based feature attributions for CatBoost models
- Visualization helpers (propensity score overlap, distributions and boxplots)

## API Reference

```{eval-rst}
.. currentmodule:: causalis.eda

.. autosummary::
   :toctree: generated
   :caption: Public objects in causalis.eda
   :recursive:
   :nosignatures:

   CausalEDA
   CausalDataLite
```

### CausalEDA

```{eval-rst}
.. automodule:: causalis.eda.eda
   :members: CausalEDA
   :undoc-members:
   :show-inheritance:
```

Selected methods:

```{eval-rst}
.. currentmodule:: causalis.eda.eda

.. autosummary::
   :toctree: generated
   :nosignatures:

   CausalEDA.data_shape
   CausalEDA.outcome_stats
   CausalEDA.fit_propensity
   CausalEDA.confounders_roc_auc
   CausalEDA.positivity_check
   CausalEDA.plot_ps_overlap
   CausalEDA.confounders_means
   CausalEDA.outcome_fit
   CausalEDA.outcome_plots
   CausalEDA.treatment_features
```

### CausalDataLite

```{eval-rst}
.. currentmodule:: causalis.eda.eda

.. autoclass:: CausalDataLite
   :members:
   :show-inheritance:
```