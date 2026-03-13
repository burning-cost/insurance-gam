"""
conftest.py — Shared fixtures for insurance-anam tests.

Synthetic data generation is designed to be fast (< 1 second on CPU) and
numerically well-conditioned. All random seeds are fixed for reproducibility.

The synthetic Poisson frequency dataset has known shape functions:
- driver_age: U-shaped (young and old drivers have higher frequency)
- vehicle_age: monotone increasing (older vehicles have more claims)
- ncd: monotone decreasing (more no-claims discount = lower frequency)
- region: categorical, 4 levels
- vehicle_type: categorical, 3 levels

True data-generating process:
  log(mu) = -3.0  [base]
           + f_age(driver_age)
           + f_vage(vehicle_age)
           + f_ncd(ncd)
           + f_region(region)
           + f_vtype(vehicle_type)
           + log(exposure)
  y ~ Poisson(mu)
"""

import numpy as np
import pytest
import torch

from insurance_gam.anam.model import FeatureConfig, InteractionConfig


SEED = 42


def _true_log_mu(
    driver_age: np.ndarray,
    vehicle_age: np.ndarray,
    ncd: np.ndarray,
    region: np.ndarray,
    vehicle_type: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    """True log-mean for synthetic Poisson data."""
    # U-shaped age effect (centred at ~40)
    f_age = 0.5 * ((driver_age - 40) / 20) ** 2 - 0.3

    # Monotone increasing vehicle age effect
    f_vage = 0.3 * (vehicle_age / 10)

    # Monotone decreasing NCD effect
    f_ncd = -0.4 * (ncd / 5)

    # Region effects (4 levels: 0=base, 1=+0.2, 2=-0.1, 3=+0.3)
    region_effects = np.array([0.0, 0.2, -0.1, 0.3])
    f_region = region_effects[region]

    # Vehicle type effects (3 levels: 0=base, 1=+0.1, 2=-0.2)
    vtype_effects = np.array([0.0, 0.1, -0.2])
    f_vtype = vtype_effects[vehicle_type]

    base = -3.0
    log_mu = base + f_age + f_vage + f_ncd + f_region + f_vtype + np.log(exposure)
    return log_mu


def _generate_synthetic_insurance(
    n: int = 2000,
    seed: int = SEED,
) -> dict:
    """Generate synthetic insurance frequency dataset."""
    rng = np.random.default_rng(seed)

    driver_age = rng.uniform(18, 80, size=n).astype(np.float32)
    vehicle_age = rng.uniform(0, 15, size=n).astype(np.float32)
    ncd = rng.integers(0, 6, size=n).astype(np.float32)  # 0–5 NCD steps
    region = rng.integers(0, 4, size=n).astype(np.int32)
    vehicle_type = rng.integers(0, 3, size=n).astype(np.int32)
    exposure = rng.uniform(0.1, 1.0, size=n).astype(np.float32)

    log_mu = _true_log_mu(
        driver_age, vehicle_age, ncd, region, vehicle_type, exposure
    )
    mu = np.exp(log_mu)
    y = rng.poisson(mu).astype(np.float32)

    # Build feature matrix: continuous first, then categorical
    X = np.column_stack(
        [driver_age, vehicle_age, ncd, region.astype(np.float32), vehicle_type.astype(np.float32)]
    )

    return {
        "X": X,
        "y": y,
        "exposure": exposure,
        "driver_age": driver_age,
        "vehicle_age": vehicle_age,
        "ncd": ncd,
        "region": region,
        "vehicle_type": vehicle_type,
        "log_mu": log_mu,
        "feature_names": ["driver_age", "vehicle_age", "ncd", "region", "vehicle_type"],
        "n_categories": {"region": 4, "vehicle_type": 3},
    }


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_data():
    """Full synthetic dataset (2000 observations)."""
    return _generate_synthetic_insurance(n=2000, seed=SEED)


@pytest.fixture(scope="session")
def small_synthetic_data():
    """Small synthetic dataset for fast unit tests (200 observations)."""
    return _generate_synthetic_insurance(n=200, seed=SEED + 1)


@pytest.fixture(scope="session")
def feature_configs():
    """FeatureConfig list matching the synthetic data schema."""
    return [
        FeatureConfig(name="driver_age", feature_type="continuous", monotonicity="none"),
        FeatureConfig(name="vehicle_age", feature_type="continuous", monotonicity="increasing"),
        FeatureConfig(name="ncd", feature_type="continuous", monotonicity="decreasing"),
        FeatureConfig(name="region", feature_type="categorical", n_categories=4, embedding_dim=4),
        FeatureConfig(name="vehicle_type", feature_type="categorical", n_categories=3, embedding_dim=4),
    ]


@pytest.fixture(scope="session")
def interaction_configs():
    """InteractionConfig for driver_age x vehicle_age."""
    return [
        InteractionConfig(feature_i="driver_age", feature_j="vehicle_age")
    ]


@pytest.fixture(scope="session")
def small_anam_model(feature_configs):
    """Untrained ANAMModel for structural tests."""
    from insurance_gam.anam.model import ANAMModel
    return ANAMModel(
        feature_configs=feature_configs,
        link="log",
        hidden_sizes=[16, 8],
    )


@pytest.fixture(scope="session")
def trained_anam(synthetic_data, feature_configs):
    """Quickly trained ANAM (few epochs) for integration tests."""
    from insurance_gam.anam.model import ANAMModel
    from insurance_gam.anam.trainer import ANAMTrainer, TrainingConfig

    model = ANAMModel(
        feature_configs=feature_configs,
        link="log",
        hidden_sizes=[16, 8],
    )
    config = TrainingConfig(
        loss="poisson",
        n_epochs=5,
        batch_size=256,
        learning_rate=1e-3,
        lambda_smooth=1e-5,
        lambda_l2=1e-5,
        patience=999,
        verbose=0,
        device="cpu",
    )
    trainer = ANAMTrainer(model, config)
    data = synthetic_data
    trainer.fit(data["X"], data["y"], exposure=data["exposure"])
    return trainer.model
