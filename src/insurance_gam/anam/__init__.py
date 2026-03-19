"""
insurance_gam.anam — Actuarial Neural Additive Model subpackage.

Re-exports the full public API of the original insurance-anam package.

Requires the ``neural`` extra::

    pip install insurance-gam[neural]
"""

try:
    from .api import ANAM
    from .feature_network import CategoricalFeatureNetwork, FeatureNetwork
    from .interaction_network import InteractionNetwork
    from .losses import (
        bernoulli_deviance,
        gamma_deviance,
        l1_sparsity_penalty,
        l2_ridge_penalty,
        poisson_deviance,
        smoothness_penalty,
        tweedie_deviance,
    )
    from .model import ANAMModel, FeatureConfig, InteractionConfig
    from .shapes import ShapeFunction, extract_shape_functions, plot_all_shapes
    from .trainer import ANAMTrainer, TrainingConfig, TrainingHistory
    from .utils import (
        StandardScaler,
        compare_shapes_to_glm,
        compute_deviance_stat,
        select_interactions_correlation,
        select_interactions_residual,
        shapes_to_relativity_table,
    )
except ImportError as _e:
    raise ImportError(
        f"insurance_gam.anam requires PyTorch. "
        f"Install it with: pip install insurance-gam[neural]\n"
        f"Original error: {_e}"
    ) from _e

__all__ = [
    "ANAM",
    "ANAMModel",
    "FeatureConfig",
    "InteractionConfig",
    "FeatureNetwork",
    "CategoricalFeatureNetwork",
    "InteractionNetwork",
    "ANAMTrainer",
    "TrainingConfig",
    "TrainingHistory",
    "poisson_deviance",
    "gamma_deviance",
    "tweedie_deviance",
    "bernoulli_deviance",
    "smoothness_penalty",
    "l1_sparsity_penalty",
    "l2_ridge_penalty",
    "ShapeFunction",
    "extract_shape_functions",
    "plot_all_shapes",
    "StandardScaler",
    "select_interactions_correlation",
    "select_interactions_residual",
    "shapes_to_relativity_table",
    "compare_shapes_to_glm",
    "compute_deviance_stat",
]
