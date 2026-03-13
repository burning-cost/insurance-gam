"""
test_model.py — Tests for ANAMModel.
"""

import numpy as np
import pytest
import torch

from insurance_gam.anam.model import ANAMModel, FeatureConfig, InteractionConfig


class TestANAMModel:
    def test_basic_forward(self, small_anam_model, anam_synthetic_data):
        """Model forward pass returns correct shape."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:32], dtype=torch.float32)
        out = model(X)
        assert out.shape == (32,)

    def test_output_positive_log_link(self, small_anam_model, anam_synthetic_data):
        """Log-link model must output strictly positive values."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:100], dtype=torch.float32)
        out = model(X)
        assert (out > 0).all()

    def test_output_bounded_logit_link(self, feature_configs, anam_synthetic_data):
        """Logit-link model must output values in (0, 1)."""
        model = ANAMModel(feature_configs=feature_configs, link="logit", hidden_sizes=[8])
        X = torch.tensor(anam_synthetic_data["X"][:50], dtype=torch.float32)
        out = model(X)
        assert (out > 0).all() and (out < 1).all()

    def test_identity_link(self, feature_configs, anam_synthetic_data):
        """Identity-link model returns unbounded values."""
        model = ANAMModel(feature_configs=feature_configs, link="identity", hidden_sizes=[8])
        X = torch.tensor(anam_synthetic_data["X"][:10], dtype=torch.float32)
        out = model(X)
        assert out.shape == (10,)

    def test_log_exposure_offset(self, small_anam_model, anam_synthetic_data):
        """Model with log_exposure=0 should give different result than without."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:10], dtype=torch.float32)
        out_no_exp = model(X, log_exposure=None)
        log_exp = torch.zeros(10)  # log(1) = 0, so same as no offset
        out_zero_exp = model(X, log_exposure=log_exp)
        # log(1) = 0, so output should be identical
        assert torch.allclose(out_no_exp, out_zero_exp, atol=1e-5)

    def test_exposure_multiplies_predictions(self, small_anam_model, anam_synthetic_data):
        """Doubling log_exposure should multiply predictions by e (for log link)."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:10], dtype=torch.float32)
        log_exp_1 = torch.ones(10) * np.log(1.0)
        log_exp_2 = torch.ones(10) * np.log(2.0)
        out_1 = model(X, log_exposure=log_exp_1)
        out_2 = model(X, log_exposure=log_exp_2)
        ratio = (out_2 / out_1).mean().item()
        assert ratio == pytest.approx(2.0, rel=1e-4)

    def test_feature_contribution_shape(self, small_anam_model, anam_synthetic_data):
        """feature_contribution() returns (batch,) tensor."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:20], dtype=torch.float32)
        contrib = model.feature_contribution(X, "driver_age")
        assert contrib.shape == (20,)

    def test_feature_importance_returns_all_features(self, small_anam_model, feature_configs):
        """feature_importance() should return all feature names."""
        importance = small_anam_model.feature_importance()
        feature_names = [cfg.name for cfg in feature_configs]
        assert set(importance.keys()) == set(feature_names)

    def test_feature_importance_positive(self, small_anam_model):
        """Importance values should be non-negative."""
        importance = small_anam_model.feature_importance()
        for val in importance.values():
            assert val >= 0.0

    def test_project_monotone_weights_no_error(self, small_anam_model):
        """Calling project_monotone_weights should not raise."""
        small_anam_model.project_monotone_weights()

    def test_monotone_increasing_enforced_after_projection(self, anam_synthetic_data):
        """Model with monotone_increasing constraint should be monotone after projection."""
        configs = [
            FeatureConfig(name="x1", feature_type="continuous", monotonicity="increasing"),
            FeatureConfig(name="x2", feature_type="continuous", monotonicity="none"),
        ]
        model = ANAMModel(feature_configs=configs, link="log", hidden_sizes=[16])
        model.project_monotone_weights()

        x_grid = torch.linspace(-2.0, 2.0, 100)
        x_zeros = torch.zeros(100)
        X = torch.stack([x_grid, x_zeros], dim=1)

        model.eval()
        with torch.no_grad():
            # Check contribution of x1 alone
            contrib = model.feature_contribution(X, "x1")
            diffs = contrib[1:] - contrib[:-1]
            assert (diffs >= -1e-5).all(), f"Monotone violation: min diff = {diffs.min().item()}"

    def test_n_features_property(self, small_anam_model, feature_configs):
        assert small_anam_model.n_features == len(feature_configs)

    def test_feature_names_property(self, small_anam_model, feature_configs):
        expected = [cfg.name for cfg in feature_configs]
        assert small_anam_model.feature_names == expected

    def test_gradient_flows_through_model(self, small_anam_model, anam_synthetic_data):
        """Backprop should work through the full model."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:16], dtype=torch.float32)
        y = torch.tensor(anam_synthetic_data["y"][:16], dtype=torch.float32)
        out = model(X)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        # Check at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_with_interaction_terms(self, feature_configs, anam_synthetic_data):
        """Model with interaction terms should work end-to-end."""
        interactions = [InteractionConfig(feature_i="driver_age", feature_j="vehicle_age")]
        model = ANAMModel(
            feature_configs=feature_configs,
            link="log",
            hidden_sizes=[8],
            interaction_configs=interactions,
        )
        X = torch.tensor(anam_synthetic_data["X"][:20], dtype=torch.float32)
        out = model(X)
        assert out.shape == (20,)
        assert (out > 0).all()

    def test_linear_predictor_shape(self, small_anam_model, anam_synthetic_data):
        """linear_predictor() should return (batch,) tensor."""
        model = small_anam_model
        X = torch.tensor(anam_synthetic_data["X"][:15], dtype=torch.float32)
        eta = model.linear_predictor(X)
        assert eta.shape == (15,)

    def test_categorical_features_handled(self, anam_synthetic_data, feature_configs):
        """Categorical features in X should be processed without error."""
        model = ANAMModel(feature_configs=feature_configs, link="log", hidden_sizes=[8])
        X = torch.tensor(anam_synthetic_data["X"][:10], dtype=torch.float32)
        out = model(X)
        assert torch.isfinite(out).all()
