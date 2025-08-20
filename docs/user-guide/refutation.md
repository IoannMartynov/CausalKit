# Refutation and robustness (causalkit.refutation)

The causalkit.refutation module provides quick checks to assess whether your estimated causal effect is plausible and robust. These utilities take your inference function (e.g., dml_ate) and a CausalData object, perturb the data (e.g., randomize outcome or treatment, subsample, etc.), re-run the estimator, and report the new estimate.

What it includes
- Placebo checks: refute_placebo_outcome, refute_placebo_treatment (expect near-zero effect and large p-value if the design is valid)
- Subset robustness: refute_subset (re-estimate on a random fraction of data)
- Orthogonality diagnostics: refute_irm_orthogonality (EIF-based diagnostic suite)
- Sensitivity analysis: sensitivity_analysis, sensitivity_analysis_set (robustness to unobserved confounding)

Return values
- Placebo/subset helpers return a small dict: {'theta': float, 'p_value': float}
- Orthogonality returns a diagnostics dict (keys include 'theta', 'oos_moment_test', 'orthogonality_derivatives', 'influence_diagnostics', 'trimming_info', 'diagnostic_conditions', 'overall_assessment')
- Sensitivity analysis returns a formatted text report

Minimal example
Assume you already have causal_data (see Inference guide for building it) and want to run a placebo outcome test using DoubleML ATE:

```
from causalkit.refutation import refute_placebo_outcome
from causalkit.inference.ate import dml_ate

# Replace outcome with random noise and re-estimate
res = refute_placebo_outcome(dml_ate, causal_data, random_state=42)
print(res)
# Example output (values will vary):
# {'theta': -0.0123, 'p_value': 0.9591}
```

Interpretation: Because the true outcome was replaced by noise, a well-specified design should yield theta close to 0 and a large p-value, indicating no spurious effect.
