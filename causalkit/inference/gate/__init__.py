"""
Group Average Treatment Effect (GATE) inference methods for causalkit.

This submodule provides methods for estimating group average treatment effects.
"""

from causalkit.inference.gate.gate_esimand import gate_esimand

__all__ = ["gate_esimand"]