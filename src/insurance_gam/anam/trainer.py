"""
trainer.py — Training loop for ANAMModel.

Design choices:
- Adam optimiser with cosine annealing LR schedule. Adam is less sensitive
  to learning rate than SGD, which matters here because we have many
  subnetworks of varying difficulty.
- Monotonicity projection is applied after every gradient step. This is the
  Dykstra algorithm: iterative projection onto the constraint set. For
  weight-sign constraints it reduces to a single clamp() call.
- Smoothness penalty is evaluated on a fixed grid over the training data
  range for each continuous feature. The grid is precomputed once before
  training and cached.
- Early stopping monitors validation deviance (not training loss) to avoid
  stopping on a batch that happened to have a high smoothness penalty.
- Exposure weights are passed into the loss as observation weights. This is
  different from class weights — the loss for a policy with 0.5 years
  exposure is down-weighted relative to a full-year policy.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import (
    gamma_deviance,
    l1_sparsity_penalty,
    l2_ridge_penalty,
    poisson_deviance,
    smoothness_penalty,
    tweedie_deviance,
)
from .model import ANAMModel, FeatureConfig


LossType = Literal["poisson", "tweedie", "gamma", "mse"]


@dataclass
class TrainingConfig:
    """Hyperparameters for the ANAM training loop.

    Parameters
    ----------
    loss:
        Distributional loss type. 'poisson' for frequency, 'tweedie' for
        pure premium, 'gamma' for severity, 'mse' for Gaussian.
    tweedie_p:
        Tweedie power parameter. Only used when loss='tweedie'.
        Typical range (1.0, 2.0). Common choices: 1.5 (compound Poisson).
    n_epochs:
        Maximum training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Initial Adam learning rate.
    lambda_smooth:
        Smoothness penalty weight. 0.0 disables smoothness regularisation.
    lambda_l1:
        L1 sparsity penalty weight. 0.0 disables.
    lambda_l2:
        L2 ridge penalty weight (weight decay).
    smooth_n_points:
        Number of grid points for smoothness penalty evaluation.
    val_fraction:
        Fraction of training data held out for early stopping.
    patience:
        Number of epochs without improvement before stopping.
    min_delta:
        Minimum improvement in validation loss to count as progress.
    verbose:
        Print training progress every `verbose` epochs. 0 = silent.
    device:
        PyTorch device string. 'cpu' or 'cuda'. Auto-detected if None.
    """

    loss: LossType = "poisson"
    tweedie_p: float = 1.5
    n_epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    lambda_smooth: float = 1e-4
    lambda_l1: float = 0.0
    lambda_l2: float = 1e-4
    smooth_n_points: int = 100
    val_fraction: float = 0.1
    patience: int = 10
    min_delta: float = 1e-6
    verbose: int = 10
    device: Optional[str] = None


@dataclass
class TrainingHistory:
    """Records per-epoch training and validation losses."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


class ANAMTrainer:
    """Manages the training loop for ANAMModel.

    Parameters
    ----------
    model:
        The ANAMModel to train. Modified in-place.
    config:
        Training hyperparameters.
    """

    def __init__(self, model: ANAMModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config

        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)
        self.history = TrainingHistory()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """Train the model.

        Parameters
        ----------
        X:
            Feature matrix, shape (n, n_features). Continuous features
            should be normalised before calling fit().
        y:
            Target vector, shape (n,). Claim counts, rates, or severities.
        exposure:
            Exposure vector, shape (n,). Policy years or similar. If None,
            uniform exposure (all 1.0) is assumed.

        Returns
        -------
        TrainingHistory
            Training and validation loss per epoch.
        """
        n = len(y)

        if exposure is None:
            exposure = np.ones(n, dtype=np.float32)

        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        exp_t = torch.tensor(exposure, dtype=torch.float32)

        # Train/val split
        n_val = max(1, int(n * self.config.val_fraction))
        perm = torch.randperm(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train, exp_train = X_t[train_idx], y_t[train_idx], exp_t[train_idx]
        X_val, y_val, exp_val = X_t[val_idx], y_t[val_idx], exp_t[val_idx]

        # Precompute feature ranges for smoothness penalty
        feature_ranges = self._compute_feature_ranges(X_train)

        # DataLoader
        train_dataset = TensorDataset(X_train, y_train, exp_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Optimiser and LR scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.0,  # L2 handled manually for flexibility
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.n_epochs, eta_min=self.config.learning_rate * 0.01
        )

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(self.config.n_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(
                train_loader, optimizer, feature_ranges
            )
            val_loss = self._evaluate(X_val, y_val, exp_val)

            scheduler.step()

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.epoch_times.append(time.time() - t0)

            if self.config.verbose > 0 and (epoch + 1) % self.config.verbose == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{self.config.n_epochs} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.history.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if self.config.verbose > 0:
                        print(f"Early stopping at epoch {epoch + 1}.")
                    self.history.stopped_early = True
                    break

        # Restore best weights
        self.model.load_state_dict(best_state)
        return self.history

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        feature_ranges: Dict[str, Tuple[float, float]],
    ) -> float:
        """One pass through the training data."""
        self.model.train()
        total_loss = 0.0
        total_weight = 0.0

        for X_batch, y_batch, exp_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            exp_batch = exp_batch.to(self.device)

            optimizer.zero_grad()

            # Log exposure offset for log-link models
            log_exp = torch.log(exp_batch.clamp(min=1e-8)) if self.model.link == "log" else None

            # Forward pass
            y_pred = self.model(X_batch, log_exposure=log_exp)

            # Distributional loss
            loss = self._distributional_loss(y_pred, y_batch, exp_batch)

            # Smoothness penalty over all continuous features
            if self.config.lambda_smooth > 0.0:
                for cfg in self.model.feature_configs:
                    if cfg.feature_type == "continuous":
                        x_min, x_max = feature_ranges[cfg.name]
                        net = self.model.feature_nets[cfg.name]
                        loss = loss + smoothness_penalty(
                            net, x_min, x_max,
                            n_points=self.config.smooth_n_points,
                            lambda_smooth=self.config.lambda_smooth,
                        )

            # L2 ridge
            if self.config.lambda_l2 > 0.0:
                all_nets = list(self.model.feature_nets.values()) + list(
                    self.model.interaction_nets.values()
                )
                loss = loss + l2_ridge_penalty(all_nets, self.config.lambda_l2)

            # L1 sparsity
            if self.config.lambda_l1 > 0.0:
                all_nets = list(self.model.feature_nets.values())
                loss = loss + l1_sparsity_penalty(all_nets, self.config.lambda_l1)

            loss.backward()
            optimizer.step()

            # Monotonicity projection (Dykstra step)
            self.model.project_monotone_weights()

            batch_weight = exp_batch.sum().item()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight

        return total_loss / max(total_weight, 1e-8)

    def _evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        exposure: torch.Tensor,
    ) -> float:
        """Evaluate distributional loss on a dataset (no regularisation)."""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            exposure = exposure.to(self.device)

            log_exp = torch.log(exposure.clamp(min=1e-8)) if self.model.link == "log" else None
            y_pred = self.model(X, log_exposure=log_exp)
            loss = self._distributional_loss(y_pred, y, exposure)

        return float(loss.item())

    def _distributional_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted distributional loss."""
        cfg = self.config
        if cfg.loss == "poisson":
            return poisson_deviance(y_pred, y_true, weights)
        elif cfg.loss == "tweedie":
            return tweedie_deviance(y_pred, y_true, p=cfg.tweedie_p, weights=weights)
        elif cfg.loss == "gamma":
            return gamma_deviance(y_pred, y_true, weights)
        elif cfg.loss == "mse":
            err = (y_pred - y_true) ** 2
            return (weights * err).sum() / weights.sum().clamp(min=1e-8)
        else:
            raise ValueError(f"Unknown loss: {cfg.loss!r}")

    def _compute_feature_ranges(
        self, X_train: torch.Tensor
    ) -> Dict[str, Tuple[float, float]]:
        """Compute per-feature min/max ranges from training data."""
        ranges: Dict[str, Tuple[float, float]] = {}
        for i, cfg in enumerate(self.model.feature_configs):
            if cfg.feature_type == "continuous":
                col = X_train[:, i]
                ranges[cfg.name] = (float(col.min().item()), float(col.max().item()))
        return ranges
