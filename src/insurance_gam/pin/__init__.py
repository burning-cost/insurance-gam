"""
insurance_gam.pin — Pairwise Interaction Networks subpackage.

Re-exports the full public API of the original insurance-pin package.
"""

from .model import PINModel, PINEnsemble
from .diagnostics import PINDiagnostics
from .networks import centered_hard_sigmoid

__all__ = [
    "PINModel",
    "PINEnsemble",
    "PINDiagnostics",
    "centered_hard_sigmoid",
]
