"""
insurance_gam.ebm — EBM subpackage.

Re-exports the full public API of the original insurance-ebm package,
plus EBMInsuranceWrapper (added v0.2.0) — a unified actuarial facade that
adds balance checks, interaction heatmaps, EU AI Act transparency reports,
and exposure-weighted cross-validation.

Also exports BoulevardEBM and BoulevardInference (added v0.5.1) — a
standalone boosted-stumps model with moving-average regularisation that
converges to kernel ridge regression, enabling analytical CLT-based
confidence intervals on shape functions.  No dependency on interpretML;
works on any platform with numpy, scipy, and scikit-learn.

The interpretML-based classes (InsuranceEBM, EBMInsuranceWrapper, etc.)
require the ``ebm`` extra::

    pip install insurance-gam[ebm]

BoulevardEBM and BoulevardInference are always available without extras.
"""

# Boulevard classes are always available (no heavy optional dependencies)
from ._boulevard import BoulevardEBM, BoulevardInference

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

    _HAS_INTERPRET = True

except ImportError as _e:
    _HAS_INTERPRET = False

    # Provide helpful error for classes that need interpretML
    class _InterpretRequired:
        _name: str = "unknown"

        def __init_subclass__(cls, name: str = "unknown", **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            cls._name = name

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                f"insurance_gam.ebm.{self.__class__._name} requires the "
                f"'interpret' package. Install it with: "
                f"pip install insurance-gam[ebm]\n"
                f"Original error: {_e}"
            )

    class InsuranceEBM(_InterpretRequired, name="InsuranceEBM"):  # type: ignore[no-redef]
        pass

    class RelativitiesTable(_InterpretRequired, name="RelativitiesTable"):  # type: ignore[no-redef]
        pass

    class MonotonicityEditor(_InterpretRequired, name="MonotonicityEditor"):  # type: ignore[no-redef]
        pass

    class GLMComparison(_InterpretRequired, name="GLMComparison"):  # type: ignore[no-redef]
        pass

    class EBMInsuranceWrapper(_InterpretRequired, name="EBMInsuranceWrapper"):  # type: ignore[no-redef]
        pass

    def balance_ratio(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        raise ImportError(
            f"insurance_gam.ebm.balance_ratio requires the 'interpret' package. "
            f"Install it with: pip install insurance-gam[ebm]"
        )

    def cross_validate(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        raise ImportError(
            f"insurance_gam.ebm.cross_validate requires the 'interpret' package. "
            f"Install it with: pip install insurance-gam[ebm]"
        )

    # diagnostics stub
    import types as _types
    diagnostics = _types.ModuleType("insurance_gam.ebm.diagnostics")

    def gini(*a: object, **kw: object) -> None:  # type: ignore[misc]
        raise ImportError("Requires pip install insurance-gam[ebm]")
    def lorenz_curve(*a: object, **kw: object) -> None:  # type: ignore[misc]
        raise ImportError("Requires pip install insurance-gam[ebm]")
    def double_lift(*a: object, **kw: object) -> None:  # type: ignore[misc]
        raise ImportError("Requires pip install insurance-gam[ebm]")
    def deviance(*a: object, **kw: object) -> None:  # type: ignore[misc]
        raise ImportError("Requires pip install insurance-gam[ebm]")
    def residual_plot(*a: object, **kw: object) -> None:  # type: ignore[misc]
        raise ImportError("Requires pip install insurance-gam[ebm]")
    def calibration_table(*a: object, **kw: object) -> None:  # type: ignore[misc]
        raise ImportError("Requires pip install insurance-gam[ebm]")


__all__ = [
    # Boulevard (always available)
    "BoulevardEBM",
    "BoulevardInference",
    # interpretML-dependent
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
