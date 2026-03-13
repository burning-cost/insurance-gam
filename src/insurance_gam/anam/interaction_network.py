"""
interaction_network.py — Pairwise interaction subnetworks for ANAM.

The base ANAM is purely additive: f(x) = sum_i f_i(x_i). Pairwise
interactions extend this to: f(x) = sum_i f_i(x_i) + sum_{(i,j) in S} g_{ij}(x_i, x_j)
where S is a selected set of feature pairs.

For insurance, the most important interactions are typically:
- Driver age x vehicle age (younger drivers in old vehicles)
- NCD x years licensed (correlated but distinct effects)
- Region x vehicle use (urban commuting vs rural)

Interaction pair selection is handled in utils.py. The network here just
takes two scalar inputs and outputs one scalar contribution.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class InteractionNetwork(nn.Module):
    """Pairwise interaction subnetwork g_{ij}(x_i, x_j).

    Takes two feature values and learns their joint effect. The network
    is intentionally shallow to avoid overfitting and to keep the
    interaction contribution interpretable as a 2D surface.

    Parameters
    ----------
    feature_indices:
        Tuple (i, j) identifying which features this network handles.
        Used for bookkeeping and shape function export.
    hidden_sizes:
        Width of each hidden layer. Shallower than single-feature nets
        is recommended — interactions should be simple corrections to
        the additive baseline.
    dropout:
        Dropout rate between hidden layers.
    """

    def __init__(
        self,
        feature_indices: tuple[int, int],
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 16]

        self.feature_indices = feature_indices
        self.hidden_sizes = hidden_sizes

        layers: list[nn.Module] = []
        in_dim = 2  # two features concatenated

        for i, out_dim in enumerate(hidden_sizes):
            linear = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0 and i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        output_layer = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_i, x_j:
            Shape (batch,) or (batch, 1). Values for the two interacting
            features.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1). Interaction contribution g_{ij}(x_i, x_j).
        """
        if x_i.dim() == 1:
            x_i = x_i.unsqueeze(-1)
        if x_j.dim() == 1:
            x_j = x_j.unsqueeze(-1)
        combined = torch.cat([x_i, x_j], dim=-1)  # (batch, 2)
        return self.network(combined)

    def interaction_grid(
        self,
        xi_min: float,
        xi_max: float,
        xj_min: float,
        xj_max: float,
        n_points: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate interaction surface over a 2D grid.

        Returns (xi_grid, xj_grid, g_values) where g_values is an
        (n_points, n_points) tensor representing the interaction surface.
        Useful for heatmap visualisation.
        """
        self.eval()
        with torch.no_grad():
            xi = torch.linspace(xi_min, xi_max, n_points)
            xj = torch.linspace(xj_min, xj_max, n_points)
            xi_grid, xj_grid = torch.meshgrid(xi, xj, indexing="ij")
            xi_flat = xi_grid.reshape(-1)
            xj_flat = xj_grid.reshape(-1)
            g_flat = self.forward(xi_flat, xj_flat).squeeze(-1)
            g_values = g_flat.reshape(n_points, n_points)
        return xi_grid, xj_grid, g_values
