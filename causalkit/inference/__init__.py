"""
Analysis module for causalkit.

This module provides statistical inference tools for causal inference.
"""

# Re-export commonly used functions at the package level
from causalkit.inference.att.ttest import ttest
from causalkit.inference.att.conversion_z_test import conversion_z_test
from causalkit.inference.att.bootstrap_diff_means import bootstrap_diff_means
from causalkit.inference.ate.dml_ate_source import dml_ate_source as dml
from causalkit.inference.ate.causalforestdml import causalforestdml as causalforestdml
from causalkit.inference.att.dml_att_source import dml_att_source
# Backward-compatible alias for convenience
dml_att = dml_att_source
from causalkit.inference.cate import cate_esimand
from causalkit.inference.gate import gate_esimand

# Provide a backward/alternative import path so that
# `from causalkit.inference.ttest import ttest` works.
# We alias the existing module `causalkit.inference.att.ttest` as a submodule
# of this package without duplicating implementation files.
import importlib
import sys as _sys

_ttest_module = importlib.import_module('.att.ttest', __name__)
_sys.modules[__name__ + '.ttest'] = _ttest_module

__all__ = ['ttest', 'conversion_z_test', 'bootstrap_diff_means', 'dml', 'causalforestdml', 'dml_att_source', 'cate_esimand', 'gate_esimand']