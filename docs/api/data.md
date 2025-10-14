# Data Module

The `causalis.data` module provides functions for generating synthetic data for causal inference tasks.

## Overview

This module includes functions for generating:

- A/B test data with customizable parameters
- Randomized Controlled Trial (RCT) data
- Observational data for more complex causal inference scenarios

## API Reference

```{eval-rst}
.. currentmodule:: causalis.data

.. autosummary::
   :toctree: generated
   :caption: Public objects in causalis.data
   :recursive:
   :nosignatures:

   causaldata.CausalData
   generators.generate_rct_data
```

```{eval-rst}
.. currentmodule:: causalis.data.causaldata

CausalData
----------

.. autoclass:: CausalData
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: __weakref__
```

```{eval-rst}
.. automodule:: causalis.data.generators
   :members:
   :undoc-members:
   :show-inheritance:
```