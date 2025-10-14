# Inference Module

The `causalis.inference` package provides statistical inference tools for causal analysis across several estimands:
- ATT: Average Treatment effect on the Treated
- ATE: Average Treatment Effect
- CATE: Conditional Average Treatment Effect (per-observation signals)
- GATE: Grouped Average Treatment Effects

## Overview

At a glance:
- Simple tests for A/B outcomes (t-test, two-proportion z-test)
- DoubleML-based estimators for ATE and ATT
- DoubleML-based CATE signals and GATE grouping/intervals

## API Reference (Package)

```{eval-rst}
.. currentmodule:: causalis.inference

.. autosummary::
   :toctree: generated
   :caption: Common re-exports in causalis.inference
   :nosignatures:

   ttest
   conversion_z_test
   bootstrap_diff_means
   dml_atte_source
```

## ATT Utilities

```{eval-rst}
.. automodule:: causalis.inference.atte.ttest
   :members: ttest
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: causalis.inference.atte.conversion_z_test
   :members: conversion_z_test
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: causalis.inference.atte.dml_atte
   :members: dml_atte
   :undoc-members:
   :show-inheritance:
```

## ATE Utilities

```{eval-rst}
.. automodule:: causalis.inference.ate.dml_ate
   :members: dml_ate
   :undoc-members:
   :show-inheritance:
```

## CATE Utilities

```{eval-rst}
.. automodule:: causalis.inference.cate.cate_esimand
   :members: cate_esimand
   :undoc-members:
   :show-inheritance:
```

## GATE Utilities

```{eval-rst}
.. automodule:: causalis.inference.gate.gate_esimand
   :members: gate_esimand
   :undoc-members:
   :show-inheritance:
```