"""
Average Treatment Effect (ATE) inference methods for causalkit.

This module provides methods for estimating average treatment effects.
"""

from causalkit.inference.ate.dml_ate_source import dml_ate_source
from causalkit.inference.ate.dml_ate import dml_ate

__all__ = ['dml_ate_source', 'dml_ate']