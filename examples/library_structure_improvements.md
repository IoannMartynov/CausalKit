# Improving the Structure of CausalKit Library

Based on analysis of the current library structure, here's a comprehensive plan to improve the organization, particularly for adding EDA functions and organizing inference functions by different causal effect types.

## Current Structure

The library currently has this basic structure:
```
causalkit/
├── data/
│   ├── causaldata.py
│   └── generators.py
├── design/
│   ├── mde.py
│   └── traffic_splitter.py
└── inference/
    └── ttest.py
```

## Proposed Structure Improvements

### 1. Add an EDA Module

Create a dedicated module for exploratory data analysis:

```
causalkit/
├── eda/
│   ├── __init__.py
│   ├── balance.py        # For covariate balance checks
│   ├── visualization.py  # For plotting treatment/outcome relationships
│   └── diagnostics.py    # For common causal inference diagnostics
```

The EDA module would focus on:
- Visualizing treatment-outcome relationships
- Checking covariate balance between treatment groups
- Assessing common assumptions in causal inference
- Detecting potential confounding

### 2. Restructure the Inference Module

Reorganize the inference module to separate different types of causal effects:

```
causalkit/
└── inference/
    ├── __init__.py
    ├── common.py         # Common utilities for all inference methods
    ├── ate/              # Average Treatment Effect
    │   ├── __init__.py
    │   ├── parametric.py # Parametric methods for ATE
    │   └── nonparametric.py # Nonparametric methods for ATE
    ├── att/              # Average Treatment Effect on the Treated
    │   ├── __init__.py
    │   ├── matching.py   # Matching methods for ATT
    │   └── weighting.py  # Weighting methods for ATT
    ├── cate/             # Conditional Average Treatment Effect
    │   ├── __init__.py
    │   ├── tree_based.py # Tree-based methods for CATE
    │   └── meta.py       # Meta-learners for CATE
    └── gate/             # Group Average Treatment Effect
        ├── __init__.py
        └── subgroup.py   # Subgroup analysis methods
```

### 3. Implementation Details

**For the EDA module:**

- `balance.py`: Include functions for standardized mean differences, propensity score distributions, and covariate balance tables
- `visualization.py`: Implement functions for treatment-outcome plots, propensity score distributions, and effect heterogeneity visualizations
- `diagnostics.py`: Add functions for checking positivity violations, instrumental variable validity, and parallel trends (for diff-in-diff)

**For the restructured inference module:**

- **ATE (Average Treatment Effect):**
  - Simple methods like t-tests (move your existing ttest.py here)
  - Regression adjustment methods
  - Doubly robust methods

- **ATT (Average Treatment Effect on Treated):**
  - Matching methods (propensity score, exact, coarsened exact)
  - Weighting methods (inverse probability weighting)

- **CATE (Conditional Average Treatment Effect):**
  - Tree-based methods (causal forests, Bayesian additive regression trees)
  - Meta-learners (S-learner, T-learner, X-learner, R-learner)

- **GATE (Group Average Treatment Effect):**
  - Subgroup analysis methods
  - Heterogeneous treatment effect testing

### 4. Imports and API Design

Ensure clean imports by properly structuring the `__init__.py` files:

For example, after implementing the proposed structure, the `causalkit/inference/__init__.py` file might look like this:
```python
"""
Inference module for causalkit.

This module provides statistical inference tools for causal inference,
organized by the type of causal effect being estimated.
"""

# NOTE: This is an example of how imports would look after implementing the proposed structure.
# These modules and functions don't exist yet in the current codebase.

# Import main functions from submodules
from causalkit.inference.ate import parametric_ate, nonparametric_ate
from causalkit.inference.att import matching_att, weighting_att
from causalkit.inference.cate import estimate_cate
from causalkit.inference.gate import estimate_gate

__all__ = [
    'parametric_ate', 'nonparametric_ate',
    'matching_att', 'weighting_att',
    'estimate_cate', 'estimate_gate'
]
```

### 5. Implementation Strategy

1. Start by creating the directory structure
2. Move existing code to appropriate locations (e.g., move ttest.py to the ATE submodule)
3. Implement new functionality incrementally, starting with the most commonly used methods
4. Add comprehensive tests for each new component in the tests directory
5. Update documentation to reflect the new structure

This structure will make the library more maintainable, easier to navigate, and better organized for future extensions. It also clearly separates different types of causal effects, making it more intuitive for users to find the specific methods they need.