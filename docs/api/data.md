# Data Module

The `causalkit.data` module provides functions for generating synthetic data for causal inference tasks.

## Overview

This module includes functions for generating:

- A/B test data with customizable parameters
- Randomized Controlled Trial (RCT) data
- Observational data for more complex causal inference scenarios

## API Reference

::: causalkit.data.generators
    handler: python
    selection:
      members:
        - generate_ab_test_data
        - generate_rct_data
        - generate_obs_data
    rendering:
      show_root_heading: false
      show_source: true