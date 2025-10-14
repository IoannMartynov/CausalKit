"""
Conditional Average Treatment Effect (CATE) inference methods for causalis.

This submodule provides methods for estimating conditional average treatment effects.
"""

from causalis.inference.cate.cate_esimand import cate_esimand

__all__ = ["cate_esimand"]
