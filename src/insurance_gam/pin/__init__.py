"""
insurance_gam.pin — Pairwise Interaction Networks subpackage.

Re-exports the full public API of the original insurance-pin package.

Requires the ``pin`` extra::

    pip install insurance-gam[pin]
"""

try:
    from .model import PINModel, PINEnsemble
    from .diagnostics import PINDiagnostics
    from .networks import centered_hard_sigmoid
except ImportError as e:
    raise ImportError(
        "insurance_gam.pin requires the pin extra. "
        "Install with: pip install insurance-gam[pin]\n"
        f"Original error: {e}"
    ) from e

__all__ = [
    "PINModel",
    "PINEnsemble",
    "PINDiagnostics",
    "centered_hard_sigmoid",
]
