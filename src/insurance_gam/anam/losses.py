"""
losses.py — Distributional loss functions and regularisation penalties.

Insurance claim frequency follows Poisson (or negative binomial) and pure
premium follows Tweedie (Poisson claim count x Gamma severity). These are
not arbitrary choices — they reflect the data-generating process.

Using deviance rather than MSE matters because:
1. Deviance is the proper scoring rule for exponential family distributions.
2. Poisson deviance penalises under-prediction more than over-prediction
   (asymmetric loss), which aligns with reserve adequacy concerns.
3. Tweedie deviance handles the zero-inflation naturally — no need to
   two-part models unless you specifically want them.

Smoothness penalty:
The second-order difference penalty discourages the shape function from
having sharp local spikes. Applied by evaluating f_i at a grid of sorted
feature values and penalising sum((f_{k+2} - 2*f_{k+1} + f_k)^2).

All loss functions return mean loss (not sum) so gradients scale
consistently regardless of batch size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Distributional losses
# ---------------------------------------------------------------------------


def poisson_deviance(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Poisson deviance loss (mean over batch).

    Deviance = 2 * w * [y * log(y / mu) - (y - mu)]
    where mu = y_pred, y = y_true, w = observation weight.

    Parameters
    ----------
    y_pred:
        Predicted mean (mu), strictly positive. Shape (batch,).
    y_true:
        Observed values. Shape (batch,).
    weights:
        Observation weights (e.g. exposure). Shape (batch,). If None,
        uses uniform weights.
    eps:
        Numerical floor for y_true in log computation.

    Returns
    -------
    torch.Tensor
        Scalar mean deviance.
    """
    mu = y_pred.clamp(min=eps)
    y = y_true.clamp(min=eps)

    # y * log(y/mu) term: undefined at y=0, use 0 * log(0) = 0 convention
    log_term = torch.where(
        y_true > eps,
        y * (torch.log(y) - torch.log(mu)),
        torch.zeros_like(y),
    )
    d = 2.0 * (log_term - (y_true - mu))

    if weights is not None:
        return (weights * d).sum() / weights.sum().clamp(min=eps)
    return d.mean()


def gamma_deviance(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gamma deviance loss (mean over batch).

    Deviance = 2 * w * [log(mu/y) + (y - mu)/mu]
    where mu = y_pred, y = y_true.

    Used for claim severity (positive, right-skewed).
    """
    mu = y_pred.clamp(min=eps)
    y = y_true.clamp(min=eps)

    d = 2.0 * (torch.log(mu / y) + (y - mu) / mu)

    if weights is not None:
        return (weights * d).sum() / weights.sum().clamp(min=eps)
    return d.mean()


def tweedie_deviance(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    p: float = 1.5,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Tweedie deviance loss (mean over batch).

    For p in (1, 2):
        D(y, mu) = 2 * [y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)]

    Special cases:
    - p=1: Poisson (use poisson_deviance for numerical stability)
    - p=2: Gamma (use gamma_deviance)
    - p=1.5: Inverse Gaussian-like, common for pure premium

    Parameters
    ----------
    p:
        Tweedie power parameter. Must not equal 1 or 2. Typical range
        (1.0, 2.0) for compound Poisson-Gamma.
    """
    if abs(p - 1.0) < 1e-6:
        return poisson_deviance(y_pred, y_true, weights, eps)
    if abs(p - 2.0) < 1e-6:
        return gamma_deviance(y_pred, y_true, weights, eps)

    mu = y_pred.clamp(min=eps)
    y = y_true.clamp(min=eps)

    term1 = y.pow(2 - p) / ((1 - p) * (2 - p))
    term2 = y * mu.pow(1 - p) / (1 - p)
    term3 = mu.pow(2 - p) / (2 - p)

    # Handle y=0 in term1: 0^(2-p) for p<2 is 0, so term1 -> 0
    term1 = torch.where(y_true < eps, torch.zeros_like(term1), term1)

    d = 2.0 * (term1 - term2 + term3)

    if weights is not None:
        return (weights * d).sum() / weights.sum().clamp(min=eps)
    return d.mean()


def bernoulli_deviance(
    y_pred_logit: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary cross-entropy deviance (logit inputs).

    For binary outcomes (lapse, catastrophic event indicators).
    y_pred_logit is the raw network output (before sigmoid).
    """
    d = F.binary_cross_entropy_with_logits(y_pred_logit, y_true, reduction="none")
    if weights is not None:
        return (weights * d).sum() / weights.sum().clamp(min=eps)
    return d.mean()


# ---------------------------------------------------------------------------
# Regularisation penalties
# ---------------------------------------------------------------------------


def smoothness_penalty(
    feature_network: "torch.nn.Module",
    x_min: float,
    x_max: float,
    n_points: int = 100,
    lambda_smooth: float = 1e-4,
) -> torch.Tensor:
    """Second-order difference penalty on a feature network's shape.

    Evaluates f_i at n_points evenly spaced over [x_min, x_max] and
    penalises second differences: sum((f_{k+2} - 2*f_{k+1} + f_k)^2).

    This discourages shape functions that change direction rapidly.
    Lambda_smooth controls the trade-off between fit and smoothness.
    """
    x_grid = torch.linspace(x_min, x_max, n_points)
    f_vals = feature_network(x_grid).squeeze(-1)  # (n_points,)

    # Second-order differences: f[k+2] - 2*f[k+1] + f[k]
    second_diff = f_vals[2:] - 2 * f_vals[1:-1] + f_vals[:-2]
    penalty = lambda_smooth * (second_diff ** 2).sum()
    return penalty


def l1_sparsity_penalty(
    feature_networks: list["torch.nn.Module"],
    lambda_l1: float = 1e-5,
) -> torch.Tensor:
    """L1 penalty on output layer weights of each subnetwork.

    Encourages some subnetworks to output near-zero (feature selection).
    Applied only to the output layer to avoid over-shrinking intermediate
    representations.
    """
    penalty = torch.tensor(0.0)
    for net in feature_networks:
        for name, param in net.named_parameters():
            # Target the final linear layer weights
            if "weight" in name:
                penalty = penalty + param.abs().sum()
    return lambda_l1 * penalty


def l2_ridge_penalty(
    feature_networks: list["torch.nn.Module"],
    lambda_l2: float = 1e-4,
) -> torch.Tensor:
    """L2 ridge penalty across all subnetwork weights.

    Standard weight decay. Stabilises training especially when many
    subnetworks sum together — without it, individual nets can grow
    large while cancelling each other out.
    """
    penalty = torch.tensor(0.0)
    for net in feature_networks:
        for param in net.parameters():
            penalty = penalty + (param ** 2).sum()
    return lambda_l2 * penalty
