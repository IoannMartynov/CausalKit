# Analysis Module

The `causalkit.analysis` module provides statistical analysis tools for causal inference.

## Overview

This module includes functions for:

- Comparing A/B test results using two-sample t-tests
- Analyzing experimental data using OLS regression with treatment dummies
- Advanced analysis using Partial Linear Regression (PLR) with DoubleML

## API Reference

::: causalkit.analysis.ttest
    handler: python
    selection:
      members:
        - compare_ab
        - compare_ab_with_plr
    rendering:
      show_root_heading: false
      show_source: true