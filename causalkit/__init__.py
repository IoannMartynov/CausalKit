"""
CausalKit: A Python package for causal inference.
"""

import warnings

# Suppress noisy tqdm warning in environments without ipywidgets
try:
    from tqdm import TqdmWarning  # type: ignore
    warnings.filterwarnings(
        "ignore",
        message=r".*IProgress not found.*",
        category=TqdmWarning,
    )
except Exception:
    # If tqdm is not installed or any issue arises, do not fail import
    pass

from causalkit import data
from causalkit import design
from causalkit import inference

__version__ = "0.1.0"
__all__ = ["data", "design", "analysis"]
