"""
model.py — ANAMModel: orchestrates per-feature subnetworks into a full model.

The model computes:
    eta = sum_i f_i(x_i) + sum_{(i,j) in interactions} g_{ij}(x_i, x_j) + b
    mu = link_inverse(eta + log(exposure))  [for log link]

where b is a scalar bias (learnable), f_i are FeatureNetworks, g_{ij} are
InteractionNetworks, and link_inverse converts the linear predictor to a
predicted mean.

The exposure offset is handled as: eta + log(exposure) before the link
function, exactly as in GLMs. This ensures predictions are on a per-unit
(per-policy-year) basis when the target is claim counts, not claim rates.

Feature types supported:
- 'continuous': standard FeatureNetwork on normalised scalar input
- 'categorical': CategoricalFeatureNetwork with embedding

Column ordering matches the order features are declared at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .feature_network import CategoricalFeatureNetwork, FeatureNetwork, MonotonicityDirection
from .interaction_network import InteractionNetwork


LinkFunction = Literal["log", "identity", "logit"]


@dataclass
class FeatureConfig:
    """Configuration for a single feature.

    Parameters
    ----------
    name:
        Feature name (used in output tables and plots).
    feature_type:
        'continuous' or 'categorical'.
    monotonicity:
        Monotonicity constraint. Only applies to continuous features.
    n_categories:
        Number of categories (required if feature_type='categorical').
    embedding_dim:
        Embedding dimension for categorical features.
    hidden_sizes:
        Hidden layer widths for this feature's subnetwork.
    """

    name: str
    feature_type: Literal["continuous", "categorical"] = "continuous"
    monotonicity: MonotonicityDirection = "none"
    n_categories: Optional[int] = None
    embedding_dim: int = 4
    hidden_sizes: Optional[List[int]] = None


@dataclass
class InteractionConfig:
    """Configuration for a pairwise interaction subnetwork.

    Parameters
    ----------
    feature_i, feature_j:
        Names of the two interacting features. Must be in the feature list.
    hidden_sizes:
        Hidden layer widths for the interaction subnetwork.
    """

    feature_i: str
    feature_j: str
    hidden_sizes: Optional[List[int]] = None


class ANAMModel(nn.Module):
    """Actuarial Neural Additive Model.

    Orchestrates one subnetwork per feature plus optional pairwise
    interaction networks. The output is:

        eta = bias + sum_i f_i(x_i) + sum_{(i,j)} g_{ij}(x_i, x_j)
        mu  = link_inverse(eta + log_offset)

    Parameters
    ----------
    feature_configs:
        Ordered list of FeatureConfig objects. The column order in X arrays
        passed to forward() must match this list.
    link:
        Link function. 'log' for Poisson/Tweedie/Gamma, 'identity' for
        Gaussian, 'logit' for binary.
    interaction_configs:
        Optional list of InteractionConfig for pairwise terms.
    hidden_sizes:
        Default hidden sizes for all subnetworks (overridden per-feature
        by FeatureConfig.hidden_sizes).
    dropout:
        Dropout rate applied within subnetworks.
    """

    def __init__(
        self,
        feature_configs: List[FeatureConfig],
        link: LinkFunction = "log",
        interaction_configs: Optional[List[InteractionConfig]] = None,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feature_configs = feature_configs
        self.link = link
        self.interaction_configs = interaction_configs or []

        # Map feature names to column indices
        self.feature_name_to_idx: Dict[str, int] = {
            cfg.name: i for i, cfg in enumerate(feature_configs)
        }

        default_hidden = hidden_sizes or [64, 32]

        # Build per-feature subnetworks
        feature_nets: Dict[str, nn.Module] = {}
        for cfg in feature_configs:
            net_hidden = cfg.hidden_sizes or default_hidden
            if cfg.feature_type == "continuous":
                feature_nets[cfg.name] = FeatureNetwork(
                    hidden_sizes=net_hidden,
                    monotonicity=cfg.monotonicity,
                    dropout=dropout,
                )
            elif cfg.feature_type == "categorical":
                if cfg.n_categories is None:
                    raise ValueError(
                        f"Feature '{cfg.name}' is categorical but n_categories is not set."
                    )
                feature_nets[cfg.name] = CategoricalFeatureNetwork(
                    n_categories=cfg.n_categories,
                    embedding_dim=cfg.embedding_dim,
                    hidden_sizes=net_hidden,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown feature_type: {cfg.feature_type!r}")

        self.feature_nets = nn.ModuleDict(feature_nets)

        # Build interaction subnetworks
        interaction_nets: Dict[str, nn.Module] = {}
        for icfg in self.interaction_configs:
            key = f"{icfg.feature_i}_x_{icfg.feature_j}"
            i_idx = self.feature_name_to_idx[icfg.feature_i]
            j_idx = self.feature_name_to_idx[icfg.feature_j]
            i_hidden = icfg.hidden_sizes or [32, 16]
            interaction_nets[key] = InteractionNetwork(
                feature_indices=(i_idx, j_idx),
                hidden_sizes=i_hidden,
                dropout=dropout,
            )

        self.interaction_nets = nn.ModuleDict(interaction_nets)

        # Scalar bias (learnable)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        X: torch.Tensor,
        log_exposure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all subnetworks.

        Parameters
        ----------
        X:
            Feature matrix, shape (batch, n_features). Continuous features
            should be pre-normalised. Categorical features should be integer
            indices (will be cast to long internally).
        log_exposure:
            Log of exposure (e.g. log policy duration in years), shape
            (batch,). Added as offset to the linear predictor before the
            link function. If None, no offset applied (equivalent to
            exposure=1 for all observations).

        Returns
        -------
        torch.Tensor
            Predicted means mu, shape (batch,).
        """
        batch_size = X.shape[0]

        # Accumulate linear predictor starting from bias
        eta = self.bias.expand(batch_size)

        # Feature subnetwork contributions
        for i, cfg in enumerate(self.feature_configs):
            x_i = X[:, i]
            net = self.feature_nets[cfg.name]

            if cfg.feature_type == "categorical":
                contrib = net(x_i.long()).squeeze(-1)
            else:
                contrib = net(x_i.float()).squeeze(-1)

            eta = eta + contrib

        # Interaction subnetwork contributions
        for icfg in self.interaction_configs:
            key = f"{icfg.feature_i}_x_{icfg.feature_j}"
            i_idx = self.feature_name_to_idx[icfg.feature_i]
            j_idx = self.feature_name_to_idx[icfg.feature_j]

            x_i = X[:, i_idx].float()
            x_j = X[:, j_idx].float()

            contrib = self.interaction_nets[key](x_i, x_j).squeeze(-1)
            eta = eta + contrib

        # Exposure offset
        if log_exposure is not None:
            eta = eta + log_exposure

        # Apply link inverse
        return self._link_inverse(eta)

    def _link_inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """Apply inverse link function to linear predictor."""
        if self.link == "log":
            return torch.exp(eta)
        elif self.link == "identity":
            return eta
        elif self.link == "logit":
            return torch.sigmoid(eta)
        else:
            raise ValueError(f"Unknown link function: {self.link!r}")

    def linear_predictor(
        self,
        X: torch.Tensor,
        log_exposure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return eta (linear predictor) without applying link inverse.

        Useful for inspecting additive contributions before exponentiation.
        """
        batch_size = X.shape[0]
        eta = self.bias.expand(batch_size)

        for i, cfg in enumerate(self.feature_configs):
            x_i = X[:, i]
            net = self.feature_nets[cfg.name]
            if cfg.feature_type == "categorical":
                contrib = net(x_i.long()).squeeze(-1)
            else:
                contrib = net(x_i.float()).squeeze(-1)
            eta = eta + contrib

        for icfg in self.interaction_configs:
            key = f"{icfg.feature_i}_x_{icfg.feature_j}"
            i_idx = self.feature_name_to_idx[icfg.feature_i]
            j_idx = self.feature_name_to_idx[icfg.feature_j]
            contrib = self.interaction_nets[key](
                X[:, i_idx].float(), X[:, j_idx].float()
            ).squeeze(-1)
            eta = eta + contrib

        if log_exposure is not None:
            eta = eta + log_exposure

        return eta

    def feature_contribution(
        self, X: torch.Tensor, feature_name: str
    ) -> torch.Tensor:
        """Return the contribution of a single feature for each observation.

        Useful for explaining individual predictions: the contribution from
        feature i is exactly f_i(x_i), the output of that subnetwork.

        Parameters
        ----------
        X:
            Feature matrix (batch, n_features).
        feature_name:
            Name of the feature to inspect.

        Returns
        -------
        torch.Tensor
            Shape (batch,). Values of f_i(x_i) for each row.
        """
        idx = self.feature_name_to_idx[feature_name]
        cfg = self.feature_configs[idx]
        net = self.feature_nets[feature_name]
        x_i = X[:, idx]

        if cfg.feature_type == "categorical":
            return net(x_i.long()).squeeze(-1)
        else:
            return net(x_i.float()).squeeze(-1)

    def project_monotone_weights(self) -> None:
        """Enforce monotonicity constraints on all relevant subnetworks.

        Call this after optimizer.step() in the training loop.
        Does nothing for non-monotone and categorical features.
        """
        for cfg in self.feature_configs:
            if cfg.feature_type == "continuous" and cfg.monotonicity != "none":
                net = self.feature_nets[cfg.name]
                assert isinstance(net, FeatureNetwork)
                net.project_weights()

    def feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance as the L2 norm of output layer weights.

        Larger norm = larger potential contribution from that feature. This
        is a quick heuristic for feature selection — not a replacement for
        proper permutation importance or SHAP values on the additive model.

        Returns
        -------
        Dict[str, float]
            Feature name -> importance score, sorted descending.
        """
        importances: Dict[str, float] = {}

        for name, net in self.feature_nets.items():
            total_norm = 0.0
            for param in net.parameters():
                total_norm += param.data.norm(2).item() ** 2
            importances[name] = float(total_norm ** 0.5)

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    @property
    def n_features(self) -> int:
        """Number of features this model was built for."""
        return len(self.feature_configs)

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names."""
        return [cfg.name for cfg in self.feature_configs]
