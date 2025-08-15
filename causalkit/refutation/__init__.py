"""
Refutation and robustness utilities for CausalKit.

Each helper takes an `inference_fn` (any function that accepts a
`CausalData` instance and returns a dictionary with keys
``coefficient`` and ``p_value`` – this matches all built-in
CausalKit inference routines) as well as the original `CausalData`.
The helpers modify the data (shuffle, subsample, …), re-run the
inference, and return only the new point estimate (theta) and p-value.
"""

from .placebo import (
    refute_placebo_outcome,
    refute_placebo_treatment,
    refute_subset,
)

__all__ = [
    "refute_placebo_outcome",
    "refute_placebo_treatment",
    "refute_subset",
]