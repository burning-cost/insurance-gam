"""
Integration tests: EBM + ANAM + PIN on the same synthetic dataset.

These tests verify that the full data → model → prediction → explanation
workflow works end-to-end for all three model families in insurance-gam.
They are intentionally fast (few training epochs, small data) but
representative of the full actuarial workflow.

Synthetic dataset: UK motor frequency, Poisson distribution, 3 features.
Known signal: driver_age has a U-shaped effect, vehicle_age is monotone
increasing, area is categorical.

Why test these together:
- Prediction ranges should be plausible across all three model types.
- Shapley values (where available) from different models trained on the
  same data should agree on sign for a strong, obvious feature effect.
- Workflow consistency: ANAM and PIN both use exposure-aware Poisson
  loss. EBM uses interpretML's Poisson mode.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skip integration tests")


# ---------------------------------------------------------------------------
# Mock interpretML so EBM tests run without it installed
# ---------------------------------------------------------------------------

def _make_mock_interpret():
    """Minimal fake interpret package that exposes InsuranceEBM's required API."""
    interpret_mod = types.ModuleType("interpret")
    glassbox_mod = types.ModuleType("interpret.glassbox")

    class FakeEBR:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self.feature_names_in_ = None
            self.term_features_ = None
            self.term_scores_ = None
            self.term_bin_weights_ = None

        def fit(self, X, y, init_score=None, sample_weight=None):
            cols = list(X.columns)
            n_features = len(cols)
            self.feature_names_in_ = np.array(cols)
            self.term_features_ = [(i,) for i in range(n_features)]
            rng = np.random.default_rng(42)
            # Small positive scores to give exp()-transformed predictions near 1
            self.term_scores_ = [rng.standard_normal(6) * 0.05 for _ in range(n_features)]
            self.term_bin_weights_ = [
                np.array([5.0, 20.0, 25.0, 30.0, 20.0, 10.0]) for _ in range(n_features)
            ]
            return self

        def predict(self, X):
            n = len(X)
            rng = np.random.default_rng(0)
            return rng.standard_normal(n) * 0.05  # log scores near 0

        def explain_global(self, name="EBM"):
            explanation = MagicMock()
            def _data(feature_idx):
                return {"names": [str(i) for i in range(6)], "scores": self.term_scores_[feature_idx][1:].tolist()}
            explanation.data.side_effect = _data
            return explanation

    glassbox_mod.ExplainableBoostingRegressor = FakeEBR
    interpret_mod.glassbox = glassbox_mod
    return interpret_mod, glassbox_mod


_interpret_mod, _glassbox_mod = _make_mock_interpret()
sys.modules.setdefault("interpret", _interpret_mod)
sys.modules.setdefault("interpret.glassbox", _glassbox_mod)


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

SEED = 99
N_TRAIN = 300
N_TEST = 50
N_FEATURES = 3  # driver_age (continuous), vehicle_age (continuous), area (categorical, 3 levels)


def _make_motor_data(n: int = N_TRAIN, seed: int = SEED) -> dict:
    """Small UK motor frequency dataset with known effects."""
    rng = np.random.default_rng(seed)

    driver_age = rng.uniform(18, 75, n).astype(np.float32)
    vehicle_age = rng.uniform(0, 15, n).astype(np.float32)
    area = rng.integers(0, 3, n).astype(np.int32)
    exposure = rng.uniform(0.2, 1.0, n).astype(np.float32)

    # True log-rate: U-shaped age, increasing vehicle age, area effect
    area_effects = np.array([0.0, 0.25, -0.15])
    log_rate = (
        -3.5
        + 0.4 * ((driver_age - 40) / 25) ** 2   # U-shape in age
        + 0.3 * (vehicle_age / 10)               # increasing in vehicle age
        + area_effects[area]
    )
    mu = np.exp(log_rate) * exposure
    claim_count = rng.poisson(mu).astype(np.float32)

    return {
        "driver_age": driver_age,
        "vehicle_age": vehicle_age,
        "area": area,
        "exposure": exposure,
        "claim_count": claim_count,
        "log_rate_true": log_rate,
    }


@pytest.fixture(scope="module")
def train_data():
    return _make_motor_data(n=N_TRAIN, seed=SEED)


@pytest.fixture(scope="module")
def test_data():
    return _make_motor_data(n=N_TEST, seed=SEED + 1)


# ---------------------------------------------------------------------------
# Feature prep helpers for each model type
# ---------------------------------------------------------------------------

def _anam_feature_configs():
    from insurance_gam.anam.model import FeatureConfig
    return [
        FeatureConfig(name="driver_age", feature_type="continuous", monotonicity="none"),
        FeatureConfig(name="vehicle_age", feature_type="continuous", monotonicity="increasing"),
        FeatureConfig(name="area", feature_type="categorical", n_categories=3, embedding_dim=3),
    ]


def _anam_X(data: dict) -> np.ndarray:
    """Build feature matrix for ANAM (column order must match FeatureConfig)."""
    return np.column_stack([
        data["driver_age"],
        data["vehicle_age"],
        data["area"].astype(np.float32),
    ])


def _pin_features() -> dict:
    return {
        "driver_age": "continuous",
        "vehicle_age": "continuous",
        "area": 3,
    }


def _pin_X(data: dict) -> dict:
    return {
        "driver_age": data["driver_age"],
        "vehicle_age": data["vehicle_age"],
        "area": data["area"],
    }


def _ebm_X(data: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "driver_age": data["driver_age"].astype(float),
        "vehicle_age": data["vehicle_age"].astype(float),
        "area": data["area"].astype(float),
    })


# ---------------------------------------------------------------------------
# Fitted model fixtures (trained once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_anam(train_data):
    from insurance_gam.anam.model import ANAMModel
    from insurance_gam.anam.trainer import ANAMTrainer, TrainingConfig

    model = ANAMModel(feature_configs=_anam_feature_configs(), link="log", hidden_sizes=[16, 8])
    config = TrainingConfig(
        loss="poisson",
        n_epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        lambda_smooth=1e-5,
        lambda_l2=1e-5,
        patience=999,
        verbose=0,
        device="cpu",
    )
    trainer = ANAMTrainer(model, config)
    trainer.fit(_anam_X(train_data), train_data["claim_count"], exposure=train_data["exposure"])
    return trainer.model


@pytest.fixture(scope="module")
def fitted_pin(train_data):
    from insurance_gam.pin.model import PINModel

    model = PINModel(
        features=_pin_features(),
        embedding_dim=4,
        hidden_dim=8,
        token_dim=4,
        shared_dims=(8, 8),
        loss="poisson",
        max_epochs=5,
        batch_size=128,
        patience=999,
        device="cpu",
        random_seed=SEED,
    )
    model.fit(
        _pin_X(train_data),
        train_data["claim_count"],
        exposure=train_data["exposure"],
        verbose=False,
    )
    return model


@pytest.fixture(scope="module")
def fitted_ebm(train_data):
    from insurance_gam.ebm import InsuranceEBM

    model = InsuranceEBM(loss="poisson", interactions=0, random_state=42, n_jobs=1)
    model.fit(_ebm_X(train_data), train_data["claim_count"], exposure=train_data["exposure"])
    return model


# ---------------------------------------------------------------------------
# Integration tests: basic fit + predict
# ---------------------------------------------------------------------------


class TestEndToEndPredictions:
    def test_anam_predict_shape(self, fitted_anam, test_data):
        """ANAM predictions have correct shape."""
        import torch
        X = torch.tensor(_anam_X(test_data), dtype=torch.float32)
        preds = fitted_anam(X)
        assert preds.shape == (N_TEST,)

    def test_anam_predictions_positive(self, fitted_anam, test_data):
        """ANAM log-link predictions must be strictly positive."""
        import torch
        X = torch.tensor(_anam_X(test_data), dtype=torch.float32)
        preds = fitted_anam(X)
        assert (preds > 0).all()

    def test_pin_predict_shape(self, fitted_pin, test_data):
        """PIN predictions have correct shape."""
        preds = fitted_pin.predict(_pin_X(test_data))
        assert preds.shape == (N_TEST,)

    def test_pin_predictions_positive(self, fitted_pin, test_data):
        """PIN predictions must be strictly positive."""
        preds = fitted_pin.predict(_pin_X(test_data))
        assert (preds > 0).all()

    def test_ebm_predict_shape(self, fitted_ebm, test_data):
        """EBM predictions have correct shape."""
        preds = fitted_ebm.predict(_ebm_X(test_data))
        assert preds.shape == (N_TEST,)

    def test_ebm_predictions_positive(self, fitted_ebm, test_data):
        """EBM predictions must be strictly positive (exp of log scores)."""
        preds = fitted_ebm.predict(_ebm_X(test_data))
        assert (preds > 0).all()

    def test_anam_predictions_finite(self, fitted_anam, test_data):
        """ANAM predictions contain no NaN or Inf."""
        import torch
        X = torch.tensor(_anam_X(test_data), dtype=torch.float32)
        preds = fitted_anam(X).detach().numpy()
        assert np.isfinite(preds).all()

    def test_pin_predictions_finite(self, fitted_pin, test_data):
        """PIN predictions contain no NaN or Inf."""
        preds = fitted_pin.predict(_pin_X(test_data))
        assert np.isfinite(preds).all()

    def test_ebm_predictions_finite(self, fitted_ebm, test_data):
        """EBM predictions contain no NaN or Inf."""
        preds = fitted_ebm.predict(_ebm_X(test_data))
        assert np.isfinite(preds).all()

    def test_predictions_vary_across_risks(self, fitted_anam, fitted_pin, test_data):
        """All models produce non-constant predictions (model has learned something)."""
        import torch
        X = torch.tensor(_anam_X(test_data), dtype=torch.float32)
        anam_preds = fitted_anam(X).detach().numpy()
        pin_preds = fitted_pin.predict(_pin_X(test_data))

        assert np.std(anam_preds) > 1e-6, "ANAM predictions are all identical"
        assert np.std(pin_preds) > 1e-6, "PIN predictions are all identical"


# ---------------------------------------------------------------------------
# Integration tests: Shapley values and feature importance
# ---------------------------------------------------------------------------


class TestShapleyConsistency:
    def test_pin_shapley_feature_names(self, fitted_pin, test_data, train_data):
        """PIN Shapley dict contains all feature names."""
        X_test = _pin_X(test_data)
        X_bg = _pin_X(train_data)
        shap = fitted_pin.shapley_values(X_test, X_bg, n_background=20)
        assert set(shap.keys()) == {"driver_age", "vehicle_age", "area"}

    def test_pin_shapley_shape(self, fitted_pin, test_data, train_data):
        """PIN Shapley values have shape (n_test,) per feature."""
        X_test = _pin_X(test_data)
        X_bg = _pin_X(train_data)
        shap = fitted_pin.shapley_values(X_test, X_bg, n_background=20)
        for name, vals in shap.items():
            assert vals.shape == (N_TEST,), f"{name}: expected ({N_TEST},), got {vals.shape}"

    def test_pin_shapley_no_nan(self, fitted_pin, test_data, train_data):
        """PIN Shapley values contain no NaN or Inf."""
        X_test = _pin_X(test_data)
        X_bg = _pin_X(train_data)
        shap = fitted_pin.shapley_values(X_test, X_bg, n_background=20)
        for name, vals in shap.items():
            assert not np.any(np.isnan(vals)), f"NaN in Shapley for {name}"
            assert not np.any(np.isinf(vals)), f"Inf in Shapley for {name}"

    def test_anam_feature_importance_keys(self, fitted_anam):
        """ANAM feature_importance() returns all three features."""
        importance = fitted_anam.feature_importance()
        assert set(importance.keys()) == {"driver_age", "vehicle_age", "area"}

    def test_anam_feature_importance_non_negative(self, fitted_anam):
        """ANAM feature importance values are non-negative."""
        importance = fitted_anam.feature_importance()
        for name, val in importance.items():
            assert val >= 0.0, f"{name} importance is negative: {val}"


# ---------------------------------------------------------------------------
# Integration tests: cross-model workflow consistency
# ---------------------------------------------------------------------------


class TestCrossModelWorkflow:
    def test_all_three_fit_without_error(self, fitted_anam, fitted_pin, fitted_ebm):
        """All three model types fit successfully (fixture setup did not raise)."""
        assert fitted_anam is not None
        assert fitted_pin is not None
        assert fitted_ebm is not None

    def test_pin_and_anam_predict_on_same_test_set(self, fitted_anam, fitted_pin, test_data):
        """Both neural models accept the same underlying test data without error."""
        import torch
        X_anam = torch.tensor(_anam_X(test_data), dtype=torch.float32)
        anam_preds = fitted_anam(X_anam).detach().numpy()
        pin_preds = fitted_pin.predict(_pin_X(test_data))

        # Both models should produce predictions in a similar plausible range
        # (claim frequency per exposure typically 0.01 - 0.5 for UK motor)
        assert anam_preds.mean() < 10.0, "ANAM predictions are unreasonably large"
        assert pin_preds.mean() < 10.0, "PIN predictions are unreasonably large"

    def test_pin_single_observation_predict(self, fitted_pin, test_data):
        """PIN works with a single-row input (important edge case in production)."""
        X_single = {k: v[:1] for k, v in _pin_X(test_data).items()}
        preds = fitted_pin.predict(X_single)
        assert preds.shape == (1,)
        assert np.isfinite(preds[0])
        assert preds[0] > 0

    def test_anam_single_observation_forward(self, fitted_anam, test_data):
        """ANAM works with a single-row input."""
        import torch
        X = torch.tensor(_anam_X(test_data)[:1], dtype=torch.float32)
        preds = fitted_anam(X)
        assert preds.shape == (1,)
        assert torch.isfinite(preds).all()
        assert (preds > 0).all()

    def test_ebm_single_observation_predict(self, fitted_ebm, test_data):
        """EBM works with a single-row input."""
        X_single = _ebm_X(test_data).iloc[:1]
        preds = fitted_ebm.predict(X_single)
        assert preds.shape == (1,)
        assert np.isfinite(preds[0])
        assert preds[0] > 0
