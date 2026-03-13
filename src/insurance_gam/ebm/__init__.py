"""
insurance_gam.ebm — EBM subpackage.

Re-exports the full public API of the original insurance-ebm package.
"""

from ._model import InsuranceEBM
from ._relativities import RelativitiesTable
from ._monotonicity import MonotonicityEditor
from ._comparison import GLMComparison
from . import _diagnostics as diagnostics

from ._diagnostics import (
    gini,
    lorenz_curve,
    double_lift,
    deviance,
    residual_plot,
    calibration_table,
)

__all__ = [
    "InsuranceEBM",
    "RelativitiesTable",
    "MonotonicityEditor",
    "GLMComparison",
    "diagnostics",
    "gini",
    "lorenz_curve",
    "double_lift",
    "deviance",
    "residual_plot",
    "calibration_table",
]
