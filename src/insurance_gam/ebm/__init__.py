"""
insurance_gam.ebm — EBM subpackage.

Re-exports the full public API of the original insurance-ebm package,
plus EBMInsuranceWrapper (added v0.2.0) — a unified actuarial facade that
adds balance checks, interaction heatmaps, EU AI Act transparency reports,
and exposure-weighted cross-validation.

Requires the ``ebm`` extra::

    pip install insurance-gam[ebm]
"""

try:
    from ._model import InsuranceEBM
    from ._relativities import RelativitiesTable
    from ._monotonicity import MonotonicityEditor
    from ._comparison import GLMComparison
    from ._wrapper import EBMInsuranceWrapper, balance_ratio, cross_validate
    from . import _diagnostics as diagnostics

    from ._diagnostics import (
        gini,
        lorenz_curve,
        double_lift,
        deviance,
        residual_plot,
        calibration_table,
    )
except ImportError as _e:
    raise ImportError(
        f"insurance_gam.ebm requires the 'interpret' package. "
        f"Install it with: pip install insurance-gam[ebm]\n"
        f"Original error: {_e}"
    ) from _e

__all__ = [
    "InsuranceEBM",
    "EBMInsuranceWrapper",
    "RelativitiesTable",
    "MonotonicityEditor",
    "GLMComparison",
    "balance_ratio",
    "cross_validate",
    "diagnostics",
    "gini",
    "lorenz_curve",
    "double_lift",
    "deviance",
    "residual_plot",
    "calibration_table",
]
