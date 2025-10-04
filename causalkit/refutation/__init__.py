"""
Refutation and robustness utilities for CausalKit.

Importing this package exposes the public functions from all refutation
submodules (overlap, score, unconfoundedness, sutva) so you can access
commonly used helpers directly via `causalkit.refutation`.
"""

# Re-export public API from subpackages
# Overlap diagnostics (public API defined in overlap/__init__.py)
from .overlap import *  # noqa: F401,F403

# Score-based refutations and diagnostics
from .score.score_validation import *  # noqa: F401,F403

# Unconfoundedness sensitivity and balance checks
from .unconfoundedness.uncofoundedness_validation import *  # noqa: F401,F403

# SUTVA helper
from .sutva.sutva_validation import *  # noqa: F401,F403

# Best-effort __all__: include common high-level entry points for discoverability.
# Note: wildcard imports above already populate the module namespace.
try:
    from .overlap import __all__ as __all_overlap  # type: ignore
except Exception:
    __all_overlap = []

try:
    from .sutva.sutva_validation import __all__ as __all_sutva  # type: ignore
except Exception:
    __all_sutva = []

# score_validation and unconfoundedness modules don't define __all__; curate a minimal list
__all_score = [
    "refute_placebo_outcome",
    "refute_placebo_treatment",
    "refute_subset",
    "refute_irm_orthogonality",
    "influence_summary",
]

__all_unconf = [
    "sensitivity_analysis",
    "get_sensitivity_summary",
    "validate_unconfoundedness_balance",
    "sensitivity_benchmark",
]

__all__ = list(dict.fromkeys([*__all_overlap, *__all_sutva, *__all_score, *__all_unconf]))