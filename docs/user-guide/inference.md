# Inference

The `causalkit.inference` module provides estimators to quantify causal effects from observational or experimental data. It offers ready‑to‑use functions for common targets such as:

- ATE (Average Treatment Effect): overall effect of a binary treatment on the outcome
- ATT (Average Treatment effect on the Treated)
- CATE/GATE: heterogeneity of effects across individuals or groups

Under the hood, ATE/ATT routines rely on Double Machine Learning (DoubleML/IRM) with sensible defaults (CatBoost learners) and cross‑fitting to reduce bias from flexible nuisance models.

## A very short ATE example

Below is the minimal flow using the DoubleML ATE estimator. It expects a `CausalData` object with one binary treatment, one outcome, and a list of confounders.

```python
from causalkit.inference.ate import dml_ate

# Assume you already constructed a CausalData object: `causal_data`
# (see the User Guide pages for data preparation and EDA)

results = dml_ate(causal_data, n_folds=3, confidence_level=0.95)

print("ATE (coefficient):", results["coefficient"])           # float
print("Std. error:", results["std_error"])                    # float
print("P-value:", results["p_value"])                         # float
print("95% CI:", results["confidence_interval"])             # (lower, upper)
```

What it returns (structure):

```text
{
  'coefficient': float,                 # estimated average treatment effect
  'std_error': float,                   # standard error
  'p_value': float,                     # p-value for H0: effect == 0
  'confidence_interval': (float, float),# (lower, upper) at the requested level
  'model': DoubleMLIRM                  # fitted DoubleML object for advanced inspection
}
```

Tip: You can pass your own ML models to `dml_ate(ml_g=..., ml_m=...)` if you want to replace the default CatBoost learners.
