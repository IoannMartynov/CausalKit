"""
Analysis module for causalkit.

This module provides statistical inference tools for causal inference.
"""

# Re-export commonly used functions at the package level
from causalkit.inference.atte.ttest import ttest
from causalkit.inference.atte.conversion_z_test import conversion_z_test
from causalkit.inference.atte.bootstrap_diff_means import bootstrap_diff_means
from causalkit.inference.atte.dml_ate_source import dml_ate_source as dml
from causalkit.inference.atte.dml_atte_source import dml_atte_source
# Backward-compatible alias for convenience
dml_att = dml_atte_source
from causalkit.inference.cate import cate_esimand
from causalkit.inference.gate import gate_esimand

# Provide a backward/alternative import path so that
# `from causalkit.inference.ttest import ttest` works.
# We alias the existing module `causalkit.inference.atte.ttest` as a submodule
# of this package without duplicating implementation files.
import importlib
import sys as _sys

_ttest_module = importlib.import_module('.atte.ttest', __name__)
_sys.modules[__name__ + '.ttest'] = _ttest_module

__all__ = ['ttest', 'conversion_z_test', 'bootstrap_diff_means', 'dml', 'dml_atte_source', 'cate_esimand', 'gate_esimand']