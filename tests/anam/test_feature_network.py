"""
test_feature_network.py — Tests for FeatureNetwork and CategoricalFeatureNetwork.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from insurance_gam.anam.feature_network import (
    CategoricalFeatureNetwork,
    FeatureNetwork,
)


class TestFeatureNetwork:
    def test_output_shape_batch(self):
        """Output should be (batch, 1) for a batch of scalar inputs."""
        net = FeatureNetwork(hidden_sizes=[16, 8])
        x = torch.randn(32)
        out = net(x)
        assert out.shape == (32, 1)

    def test_output_shape_2d_input(self):
        """Accepts (batch, 1) input directly."""
        net = FeatureNetwork(hidden_sizes=[16])
        x = torch.randn(10, 1)
        out = net(x)
        assert out.shape == (10, 1)

    def test_output_shape_single(self):
        """Single observation."""
        net = FeatureNetwork(hidden_sizes=[8])
        x = torch.tensor([1.5])
        out = net(x)
        assert out.shape == (1, 1)

    def test_gradient_flows_through_network(self):
        """Gradients must reach input for shape function training."""
        net = FeatureNetwork(hidden_sizes=[16, 8])
        x = torch.randn(10, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None

    def test_monotone_increasing_after_projection(self):
        """After projecting weights, the function should be non-decreasing."""
        torch.manual_seed(42)
        net = FeatureNetwork(hidden_sizes=[16, 8], monotonicity="increasing")
        net.project_weights()

        x_grid = torch.linspace(-3.0, 3.0, 200)
        net.eval()
        with torch.no_grad():
            f = net(x_grid).squeeze(-1)

        diffs = f[1:] - f[:-1]
        # Allow tiny numerical tolerance
        assert (diffs >= -1e-5).all(), f"Monotonicity violated: min diff = {diffs.min().item():.6f}"

    def test_monotone_decreasing_after_projection(self):
        """After projecting weights, the function should be non-increasing."""
        torch.manual_seed(42)
        net = FeatureNetwork(hidden_sizes=[16, 8], monotonicity="decreasing")
        net.project_weights()

        x_grid = torch.linspace(-3.0, 3.0, 200)
        net.eval()
        with torch.no_grad():
            f = net(x_grid).squeeze(-1)

        diffs = f[1:] - f[:-1]
        assert (diffs <= 1e-5).all(), f"Monotonicity violated: max diff = {diffs.max().item():.6f}"

    def test_no_monotone_unconstrained(self):
        """Unconstrained network should have no weight projection."""
        net = FeatureNetwork(hidden_sizes=[8], monotonicity="none")
        # Should not raise
        net.project_weights()

    def test_feature_range_returns_tensors(self):
        """feature_range() should return correct shapes."""
        net = FeatureNetwork(hidden_sizes=[8])
        x_grid, f_vals = net.feature_range(0.0, 10.0, n_points=50)
        assert x_grid.shape == (50,)
        assert f_vals.shape == (50,)

    def test_default_hidden_sizes(self):
        """Default hidden sizes [64, 32] should be used when not specified."""
        net = FeatureNetwork()
        # Check the network has layers
        layers = list(net.network.children())
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        assert len(linear_layers) == 3  # 2 hidden + 1 output

    def test_custom_hidden_sizes(self):
        """Custom hidden sizes should create the right architecture."""
        net = FeatureNetwork(hidden_sizes=[32, 16, 8])
        layers = list(net.network.children())
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        assert len(linear_layers) == 4  # 3 hidden + 1 output

    def test_dropout_layer_present(self):
        """Dropout layer should be present when rate > 0."""
        net = FeatureNetwork(hidden_sizes=[32, 16], dropout=0.1)
        has_dropout = any(isinstance(m, nn.Dropout) for m in net.network.modules())
        assert has_dropout

    def test_no_dropout_when_zero(self):
        """No Dropout layers when dropout=0."""
        net = FeatureNetwork(hidden_sizes=[32, 16], dropout=0.0)
        has_dropout = any(isinstance(m, nn.Dropout) for m in net.network.modules())
        assert not has_dropout

    def test_output_finite(self):
        """Network outputs should be finite for standard inputs."""
        net = FeatureNetwork(hidden_sizes=[16, 8])
        x = torch.linspace(-5.0, 5.0, 100)
        out = net(x)
        assert torch.isfinite(out).all()


class TestCategoricalFeatureNetwork:
    def test_output_shape(self):
        net = CategoricalFeatureNetwork(n_categories=5, embedding_dim=4, hidden_sizes=[16])
        x = torch.tensor([0, 1, 2, 3, 4])
        out = net(x)
        assert out.shape == (5, 1)

    def test_different_categories_different_outputs(self):
        """Each category should map to a distinct output (after init)."""
        torch.manual_seed(123)
        net = CategoricalFeatureNetwork(n_categories=4, embedding_dim=8, hidden_sizes=[16])
        x = torch.tensor([0, 1, 2, 3])
        out = net(x).squeeze(-1)
        # Check not all the same
        assert out.std().item() > 1e-4

    def test_category_table_keys(self):
        """category_table() should return all category indices."""
        net = CategoricalFeatureNetwork(n_categories=4)
        table = net.category_table()
        assert set(table.keys()) == {0, 1, 2, 3}

    def test_gradient_flows(self):
        net = CategoricalFeatureNetwork(n_categories=3, hidden_sizes=[8])
        x = torch.tensor([0, 1, 2])
        out = net(x)
        out.sum().backward()
        assert net.embedding.weight.grad is not None

    def test_float_input_converted_to_long(self):
        """Categorical network should handle float indices (cast to long)."""
        net = CategoricalFeatureNetwork(n_categories=4)
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        out = net(x)
        assert out.shape == (4, 1)

    def test_output_finite(self):
        net = CategoricalFeatureNetwork(n_categories=10, hidden_sizes=[16])
        x = torch.arange(10)
        out = net(x)
        assert torch.isfinite(out).all()
