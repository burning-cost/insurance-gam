"""
test_interaction_network.py — Tests for InteractionNetwork.
"""

import pytest
import torch

from insurance_gam.anam.interaction_network import InteractionNetwork


class TestInteractionNetwork:
    def test_output_shape(self):
        """Output should be (batch, 1)."""
        net = InteractionNetwork(feature_indices=(0, 1), hidden_sizes=[16, 8])
        x_i = torch.randn(20)
        x_j = torch.randn(20)
        out = net(x_i, x_j)
        assert out.shape == (20, 1)

    def test_output_with_2d_input(self):
        """Accepts (batch, 1) inputs."""
        net = InteractionNetwork(feature_indices=(0, 2), hidden_sizes=[8])
        x_i = torch.randn(10, 1)
        x_j = torch.randn(10, 1)
        out = net(x_i, x_j)
        assert out.shape == (10, 1)

    def test_gradient_flows(self):
        """Gradients should flow through the interaction network."""
        net = InteractionNetwork(feature_indices=(1, 3), hidden_sizes=[16])
        x_i = torch.randn(8, requires_grad=True)
        x_j = torch.randn(8, requires_grad=True)
        out = net(x_i, x_j)
        out.sum().backward()
        assert x_i.grad is not None
        assert x_j.grad is not None

    def test_feature_indices_stored(self):
        """Feature indices should be stored on the object."""
        net = InteractionNetwork(feature_indices=(2, 5))
        assert net.feature_indices == (2, 5)

    def test_interaction_grid_shape(self):
        """interaction_grid() should return (n, n) tensor for g_values."""
        net = InteractionNetwork(feature_indices=(0, 1), hidden_sizes=[8])
        xi_grid, xj_grid, g = net.interaction_grid(0.0, 1.0, 0.0, 1.0, n_points=20)
        assert g.shape == (20, 20)
        assert xi_grid.shape == (20, 20)
        assert xj_grid.shape == (20, 20)

    def test_output_finite(self):
        """Interaction network should produce finite outputs."""
        net = InteractionNetwork(feature_indices=(0, 1), hidden_sizes=[16])
        x_i = torch.randn(50)
        x_j = torch.randn(50)
        out = net(x_i, x_j)
        assert torch.isfinite(out).all()

    def test_default_hidden_sizes(self):
        """Default hidden sizes [32, 16] should be used when not specified."""
        import torch.nn as nn
        net = InteractionNetwork(feature_indices=(0, 1))
        linear_layers = [m for m in net.network.modules() if isinstance(m, nn.Linear)]
        # [32, 16] + output = 3 linear layers
        assert len(linear_layers) == 3

    def test_different_inputs_different_outputs(self):
        """Different (x_i, x_j) pairs should give different outputs."""
        torch.manual_seed(99)
        net = InteractionNetwork(feature_indices=(0, 1), hidden_sizes=[16])
        x_i_1 = torch.tensor([0.0])
        x_j_1 = torch.tensor([0.0])
        x_i_2 = torch.tensor([1.0])
        x_j_2 = torch.tensor([1.0])
        out_1 = net(x_i_1, x_j_1)
        out_2 = net(x_i_2, x_j_2)
        assert not torch.allclose(out_1, out_2, atol=1e-4)
