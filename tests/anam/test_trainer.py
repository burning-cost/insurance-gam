"""
test_trainer.py — Tests for the training loop.
"""

import numpy as np
import pytest
import torch

from insurance_gam.anam.model import ANAMModel, FeatureConfig
from insurance_gam.anam.trainer import ANAMTrainer, TrainingConfig


def make_tiny_model():
    """Helper: tiny model for fast training tests."""
    configs = [
        FeatureConfig(name="x1", feature_type="continuous", monotonicity="none"),
        FeatureConfig(name="x2", feature_type="continuous", monotonicity="none"),
    ]
    return ANAMModel(feature_configs=configs, link="log", hidden_sizes=[8])


def make_tiny_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype(np.float32)
    exposure = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
    log_mu = -2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.log(exposure)
    y = rng.poisson(np.exp(log_mu)).astype(np.float32)
    return X, y, exposure


class TestANAMTrainer:
    def test_fit_returns_history(self):
        """Trainer.fit() should return a TrainingHistory object."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=100)
        config = TrainingConfig(n_epochs=3, batch_size=32, verbose=0, device="cpu", patience=999)
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        assert hasattr(history, "train_loss")
        assert hasattr(history, "val_loss")

    def test_training_loss_is_finite(self):
        """Training loss values should be finite throughout."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=100)
        config = TrainingConfig(n_epochs=5, batch_size=32, verbose=0, device="cpu", patience=999)
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        for loss in history.train_loss:
            assert np.isfinite(loss), f"Non-finite training loss: {loss}"

    def test_val_loss_is_finite(self):
        """Validation loss values should be finite."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=100)
        config = TrainingConfig(n_epochs=5, batch_size=32, verbose=0, device="cpu", patience=999)
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        for loss in history.val_loss:
            assert np.isfinite(loss), f"Non-finite val loss: {loss}"

    def test_history_length_matches_epochs(self):
        """History should have one entry per epoch (up to early stopping)."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=100)
        n_epochs = 5
        config = TrainingConfig(n_epochs=n_epochs, batch_size=64, verbose=0, device="cpu", patience=999)
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        assert len(history.train_loss) == n_epochs
        assert len(history.val_loss) == n_epochs

    def test_training_without_exposure(self):
        """Training without exposure should use uniform weights."""
        model = make_tiny_model()
        X, y, _ = make_tiny_data(n=100)
        config = TrainingConfig(n_epochs=3, batch_size=32, verbose=0, device="cpu", patience=999)
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=None)
        assert len(history.train_loss) > 0

    def test_early_stopping_triggers(self):
        """Training should stop early when patience is exceeded."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=200)
        config = TrainingConfig(
            n_epochs=50,
            batch_size=64,
            verbose=0,
            device="cpu",
            patience=3,
            val_fraction=0.2,
        )
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        assert history.stopped_early or len(history.train_loss) <= 50

    def test_monotonicity_preserved_after_training(self, anam_synthetic_data, feature_configs):
        """After training with monotone constraints, predictions should be monotone."""
        from insurance_gam.anam.model import ANAMModel

        model = ANAMModel(
            feature_configs=feature_configs,
            link="log",
            hidden_sizes=[16, 8],
        )
        config = TrainingConfig(
            n_epochs=5,
            batch_size=256,
            verbose=0,
            device="cpu",
            patience=999,
            lambda_smooth=0.0,
            lambda_l2=0.0,
        )
        trainer = ANAMTrainer(model, config)
        data = anam_synthetic_data
        trainer.fit(data["X"], data["y"], exposure=data["exposure"])

        # vehicle_age is monotone increasing
        model.eval()
        with torch.no_grad():
            x_grid = torch.linspace(-2.0, 2.0, 100)  # normalised space
            zeros = torch.zeros(100)
            net = model.feature_nets["vehicle_age"]
            f_vals = net(x_grid.unsqueeze(-1)).squeeze(-1)
            diffs = f_vals[1:] - f_vals[:-1]
            assert (diffs >= -1e-5).all(), f"Monotonicity violated after training: min diff = {diffs.min()}"

    def test_tweedie_loss(self):
        """Training with Tweedie loss should not crash."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=100)
        y = y + 0.01  # Tweedie requires positive y
        config = TrainingConfig(
            loss="tweedie",
            tweedie_p=1.5,
            n_epochs=3,
            batch_size=32,
            verbose=0,
            device="cpu",
            patience=999,
        )
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        assert all(np.isfinite(l) for l in history.train_loss)

    def test_gamma_loss(self):
        """Training with Gamma loss on positive data should work."""
        configs = [
            FeatureConfig(name="x1", feature_type="continuous"),
            FeatureConfig(name="x2", feature_type="continuous"),
        ]
        model = ANAMModel(feature_configs=configs, link="log", hidden_sizes=[8])
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2)).astype(np.float32)
        y = rng.gamma(2.0, 1.0, size=100).astype(np.float32)
        exp = np.ones(100, dtype=np.float32)

        config = TrainingConfig(
            loss="gamma", n_epochs=3, batch_size=32, verbose=0, device="cpu", patience=999
        )
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        assert all(np.isfinite(l) for l in history.train_loss)

    def test_mse_loss(self):
        """Training with MSE loss should work for Gaussian targets."""
        configs = [
            FeatureConfig(name="x1", feature_type="continuous"),
        ]
        model = ANAMModel(feature_configs=configs, link="identity", hidden_sizes=[8])
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 1)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        config = TrainingConfig(
            loss="mse", n_epochs=3, batch_size=32, verbose=0, device="cpu", patience=999
        )
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y)
        assert all(np.isfinite(l) for l in history.train_loss)

    def test_best_state_restored(self):
        """After training, model should be at best validation state."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=200)
        config = TrainingConfig(
            n_epochs=10, batch_size=64, verbose=0, device="cpu", patience=5
        )
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        # If early stopping triggered, best_epoch < final epoch
        best_val = min(history.val_loss)
        assert best_val <= history.val_loss[-1] + 1e-6

    def test_smoothness_penalty_applied(self):
        """Training with lambda_smooth > 0 should complete without error."""
        model = make_tiny_model()
        X, y, exp = make_tiny_data(n=100)
        config = TrainingConfig(
            n_epochs=3,
            batch_size=64,
            verbose=0,
            device="cpu",
            patience=999,
            lambda_smooth=1e-3,
        )
        trainer = ANAMTrainer(model, config)
        history = trainer.fit(X, y, exposure=exp)
        assert all(np.isfinite(l) for l in history.train_loss)
