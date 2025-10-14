"""
Group Average Treatment Effect (GATE) inference methods for causalis.

This submodule provides methods for estimating group average treatment effects.
"""

from causalis.inference.gate.gate_esimand import gate_esimand

__all__ = ["gate_esimand"]