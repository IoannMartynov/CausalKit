# Exploratory Data Analysis (EDA)

The eda module provides quick, practical diagnostics to check whether your binary-treatment causal problem is suitable for effect estimation. It focuses on interpretability and helps you answer questions like: Do treatment and control groups overlap? Are confounders balanced? Which features drive treatment assignment and the outcome?

What you can do
- Inspect outcome by treatment: summary statistics and simple plots
- Estimate propensity scores with cross‑validation and check diagnostics (ROC AUC, positivity/overlap, score overlap plot)
- Assess covariate balance via means and standardized mean differences (SMD)
- Fit a simple outcome model (confounders only) and inspect predictive accuracy and feature attributions

Typical workflow

```python
from causalis.eda import CausalEDA
from causalis.data import CausalData  # optional import if you need to construct CausalData

# Prepare your dataset (must be CausalData or compatible with df/treatment/outcome/confounders)
causal_data = ...  # your CausalData object

eda = CausalEDA(causal_data)

# 1) Quick dataset checks
shape = eda.data_shape()  # {'n_rows': ..., 'n_columns': ...}
stats = eda.outcome_stats()  # outcome summary by treatment
fig1, fig2 = eda.outcome_plots()  # histogram + boxplot by treatment

# 2) Propensity and overlap
ps_model = eda.fit_propensity()
auc = ps_model.roc_auc  # treatment predictability
positivity = ps_model.positivity_check()
ps_model.ps_graph()  # overlap of propensity scores
shap_t = ps_model.shap  # features driving treatment

# 3) Balance
balance = eda.confounders_means()  # means, abs diff, SMD

# 4) Outcome model (confounders only)
out = eda.outcome_fit()
metrics = out.scores  # RMSE/MAE
shap_y = out.shap  # features driving outcome
```

Notes
- By default, CatBoost models are used and categorical features are handled natively; you can provide your own models if needed.
- See docs/examples/basic_example.ipynb for an end‑to‑end demonstration that uses these EDA tools before inference.
