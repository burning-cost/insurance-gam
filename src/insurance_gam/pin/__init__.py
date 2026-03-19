"""
insurance_gam.pin — Pairwise Interaction Networks subpackage.

Re-exports the full public API of the original insurance-pin package.

Requires the ``neural`` extra::

    pip install insurance-gam[neural]
"""

try:
    from .model import PINModel, PINEnsemble
    from .diagnostics import PINDiagnostics
    from .networks import centered_hard_sigmoid
except ImportError as _e:
    raise ImportError(
        f"insurance_gam.pin requires PyTorch. "
        f"Install it with: pip install insurance-gam[neural]\n"
        f"Original error: {_e}"
    ) from _e

__all__ = [
    "PINModel",
    "PINEnsemble",
    "PINDiagnostics",
    "centered_hard_sigmoid",
]
