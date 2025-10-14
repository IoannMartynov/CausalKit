"""
Analysis module for causalis.

This module provides statistical inference tools for causal inference.
"""

# Keep package import lightweight: avoid importing heavy/optional submodules at import time.
# Expose common conveniences lazily via __getattr__ to prevent ImportError when optional
# dependencies (e.g., doubleml, catboost) are not installed during documentation builds.
from typing import TYPE_CHECKING
import importlib
import sys as _sys

__all__ = [
    # Attribute-level conveniences (resolved lazily)
    'ttest', 'conversion_z_test', 'bootstrap_diff_means', 'dml_atte_source',
    # Subpackages exposed normally
    'ate', 'atte', 'cate', 'gate', 'estimators',
]

# Provide a backward/alternative import path so that
# `from causalis.inference.ttest import ttest` works without duplicate files.
# We alias the existing module `causalis.inference.atte.ttest` as a submodule of this package.
try:
    _ttest_module = importlib.import_module('.atte.ttest', __name__)
    _sys.modules[__name__ + '.ttest'] = _ttest_module
except Exception:
    # If optional dependencies inside the module are missing, skip aliasing; attribute access will still work via __getattr__.
    pass


def __getattr__(name):  # pragma: no cover
    # Lazy attribute proxying to avoid importing optional/heavy submodules at package import time
    mapping = {
        'ttest': 'atte.ttest',
        'conversion_z_test': 'atte.conversion_z_test',
        'bootstrap_diff_means': 'atte.bootstrap_diff_means',
        'dml_atte_source': 'atte.dml_atte_source',
    }
    if name in mapping:
        mod = importlib.import_module('.' + mapping[name], __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if TYPE_CHECKING:  # Static type checkers: resolve symbols
    from .atte.ttest import ttest as ttest  # noqa: F401
    from .atte.conversion_z_test import conversion_z_test as conversion_z_test  # noqa: F401
    from .atte.bootstrap_diff_means import bootstrap_diff_means as bootstrap_diff_means  # noqa: F401
    from .atte.dml_atte_source import dml_atte_source as dml_atte_source  # noqa: F401