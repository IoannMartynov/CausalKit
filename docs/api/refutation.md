# Refutation Module

The `causalis.refutation` package provides robustness and refutation utilities to stress-test causal estimates by perturbing data, checking identifying assumptions, and running sensitivity analyses.

## Overview

Key utilities:
- Placebo tests (randomize outcome or treatment, subsample)
- Sensitivity analysis for unobserved confounding (including set-based benchmarking)
- Orthogonality/IRM moment checks with out-of-sample (OOS) diagnostics

## API Reference (Package)

```{eval-rst}
.. currentmodule:: causalis.refutation

.. autosummary::
   :toctree: generated
   :caption: Public objects in causalis.refutation
   :nosignatures:

   refute_placebo_outcome
   refute_placebo_treatment
   refute_subset
   sensitivity_analysis
   get_sensitivity_summary
   refute_irm_orthogonality
```

## Placebo and Subset Refutation

```{eval-rst}
.. automodule:: causalis.refutation.score.score_validation
   :members: refute_placebo_outcome, refute_placebo_treatment, refute_subset
   :undoc-members:
   :show-inheritance:
```

## Sensitivity Analysis

```{eval-rst}
.. automodule:: causalis.refutation.unconfoundedness.uncofoundedness_validation
   :members: sensitivity_analysis, get_sensitivity_summary
   :undoc-members:
   :show-inheritance:
```

## Orthogonality Checks

```{eval-rst}
.. automodule:: causalis.refutation.score.score_validation
   :members: refute_irm_orthogonality
   :undoc-members:
   :show-inheritance:
```