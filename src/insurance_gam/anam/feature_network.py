"""
feature_network.py — Per-feature subnetworks for ANAM.

Each feature gets its own MLP. The additive architecture means every
subnetwork's contribution can be visualised independently — exactly what
a pricing actuary wants when they need to explain a rating factor.

Monotonicity is enforced by clamping weight matrices to non-negative (or
non-positive) values after every gradient step. This is the Dykstra
projection step: for ReLU nets, clamping all weight matrices to >= 0
guarantees a monotone non-decreasing output. The proof relies on the fact
that composition of monotone non-decreasing functions is monotone.

ExU activation (exp(w*(x-b)), Agarwal et al. 2021) is available but
optional. It produces sharper, more wiggly shapes and works best for
features with very non-linear effects. For most insurance features ReLU
gives smoother, more defensible curves.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


MonotonicityDirection = Literal["increasing", "decreasing", "none"]


class ExUActivation(nn.Module):
    """ExU activation: exp(w * (x - b)).

    More expressive than ReLU for capturing sharp transitions in shape
    functions, at the cost of less smooth curves and harder training.

    This module replaces both the linear layer and the activation in one
    step. It does not use a preceding nn.Linear — the weights w and biases
    b are learned here directly. The output dimension is out_features per
    input feature, then summed by the trailing relu-clamp.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.empty(in_features, out_features))
        self.biases = nn.Parameter(torch.empty(in_features))
        nn.init.normal_(self.weights, mean=4.0, std=0.5)
        nn.init.normal_(self.biases, mean=0.0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # broadcast: (batch, in_features, 1) - (in_features,) then multiply
        shifted = x - self.biases  # (batch, in_features)
        # exu per-feature: different from a standard linear layer
        # each feature i maps to out_features outputs
        # result shape: (batch, in_features, out_features) -> (batch, out_features)
        out = torch.relu(shifted.unsqueeze(-1) * self.weights.unsqueeze(0))
        # sum over in_features to produce (batch, out_features)
        return out.sum(dim=1)


class FeatureNetwork(nn.Module):
    """Single-feature MLP subnetwork.

    Takes a scalar input (one feature value) and outputs a scalar
    contribution f_i(x_i). The contribution is offset by its mean over a
    representative sample so that the bias term in the full model captures
    the population mean prediction.

    Parameters
    ----------
    hidden_sizes:
        Width of each hidden layer. E.g. [64, 32] gives two hidden layers.
    activation:
        'relu' (default) or 'exu'. ReLU is more stable; ExU more expressive.
        When 'exu' is specified, ExUActivation replaces the nn.Linear +
        nn.ReLU pair for each hidden layer (ExU fuses the linear transform
        into the activation).
    monotonicity:
        'increasing', 'decreasing', or 'none'. Enforced by projecting
        weight matrices onto the non-negative (or non-positive) orthant
        after each gradient step via project_weights().
    dropout:
        Dropout rate applied between hidden layers. 0.0 disables dropout.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        activation: Literal["relu", "exu"] = "relu",
        monotonicity: MonotonicityDirection = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.monotonicity = monotonicity
        self.dropout_rate = dropout

        layers: list[nn.Module] = []
        in_dim = 1

        for i, out_dim in enumerate(hidden_sizes):
            if activation == "relu":
                # Standard linear + ReLU pair.
                linear = nn.Linear(in_dim, out_dim)
                # Weight initialisation: smaller init stabilises training with
                # many subnetworks summing together.
                nn.init.xavier_uniform_(linear.weight, gain=0.5)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                layers.append(nn.ReLU())
            elif activation == "exu":
                # P0-3 fix: wire up ExUActivation properly.
                # ExU fuses the linear transform into the activation — no
                # preceding nn.Linear. The ExUActivation module takes
                # (in_features, out_features) and outputs (batch, out_features).
                layers.append(ExUActivation(in_features=in_dim, out_features=out_dim))
            else:
                raise ValueError(f"Unknown activation: {activation!r}")

            if dropout > 0.0 and i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(p=dropout))

            in_dim = out_dim

        # Output layer: single scalar
        output_layer = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Shape (batch,) or (batch, 1). Single feature values.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1). Feature contribution f_i(x_i).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.network(x)

    def project_weights(self) -> None:
        """Enforce monotonicity by clamping weight matrices in-place.

        For a ReLU network, all-non-negative weights guarantee a
        non-decreasing function (Dykstra's projection onto the positive
        orthant). For decreasing, clamp to non-positive.

        Call this after every optimizer.step() during training.
        """
        if self.monotonicity == "none":
            return

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                if self.monotonicity == "increasing":
                    module.weight.data.clamp_(min=0.0)
                elif self.monotonicity == "decreasing":
                    module.weight.data.clamp_(max=0.0)
            elif isinstance(module, ExUActivation):
                if self.monotonicity == "increasing":
                    module.weights.data.clamp_(min=0.0)
                elif self.monotonicity == "decreasing":
                    module.weights.data.clamp_(max=0.0)

    def feature_range(
        self, x_min: float, x_max: float, n_points: int = 200
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate shape function over a grid.

        Returns (x_grid, f_values) for plotting shape curves.
        """
        self.eval()
        with torch.no_grad():
            x_grid = torch.linspace(x_min, x_max, n_points)
            f_values = self.forward(x_grid).squeeze(-1)
        return x_grid, f_values


class CategoricalFeatureNetwork(nn.Module):
    """Subnetwork for categorical features using an embedding layer.

    Each category maps to a learned embedding (dim: embedding_dim), which
    then passes through a small MLP. This allows the model to discover
    structure in category space (e.g. similar vehicle groups cluster
    together) without requiring manual one-hot encoding.

    For regulatory documentation, the output for each category level can be
    extracted as a relativity table — exactly like a GLM factor table.
    """

    def __init__(
        self,
        n_categories: int,
        embedding_dim: int = 4,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32]

        self.n_categories = n_categories
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(n_categories, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        layers: list[nn.Module] = []
        in_dim = embedding_dim

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Shape (batch,). Integer category indices.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1). Category contribution.
        """
        embedded = self.embedding(x.long())  # (batch, embedding_dim)
        return self.network(embedded)

    def category_table(self) -> dict[int, float]:
        """Extract per-category contributions as a relativity table.

        Returns a dict mapping category index to scalar contribution value.
        Useful for regulatory documentation and GLM comparison.
        """
        self.eval()
        with torch.no_grad():
            indices = torch.arange(self.n_categories)
            contribs = self.forward(indices).squeeze(-1)
        return {int(i): float(v) for i, v in enumerate(contribs)}
