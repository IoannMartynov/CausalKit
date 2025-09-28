"""
CausalKit: A Python package for causal inference.
"""

import warnings

# Suppress noisy tqdm warning in environments without ipywidgets
try:
    from tqdm import TqdmWarning  # type: ignore
    # Apply more comprehensive filter
    warnings.filterwarnings(
        "ignore",
        message=".*IProgress not found.*",
        category=TqdmWarning,
    )
    # Also filter the exact message from the test
    warnings.filterwarnings(
        "ignore", 
        message="IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html",
        category=TqdmWarning,
    )
except Exception:
    # If tqdm is not installed or any issue arises, do not fail import
    pass

from causalkit import data
# 'design' is optional; keep import non-fatal if missing in editable installs
try:
    from causalkit import design  # type: ignore
except Exception:
    design = None  # type: ignore

__version__ = "0.1.0"
__all__ = ["data", "inference"]

# Lazily import heavy optional subpackages (e.g., inference depends on optional ML libs like catboost)
from typing import TYPE_CHECKING
import importlib

def __getattr__(name):  # pragma: no cover - behavior tested via subprocess
    if name == "inference":
        module = importlib.import_module(".inference", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if TYPE_CHECKING:  # Hint for static type checkers without importing at runtime
    from . import inference as inference  # noqa: F401
