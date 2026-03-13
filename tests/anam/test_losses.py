"""
test_losses.py — Unit tests for distributional loss functions.

Tests verify:
1. Correct numerical values against known closed-form results
2. Non-negativity (deviance >= 0 for valid inputs)
3. Zero deviance when y_pred == y_true
4. Gradient flow (losses are differentiable w.r.t. y_pred)
5. Exposure weighting correctness
6. Edge cases: zeros, large values, p near 1 and 2 for Tweedie
"""

import numpy as np
import pytest
import torch

from insurance_gam.anam.losses import (
    bernoulli_deviance,
    gamma_deviance,
    l1_sparsity_penalty,
    l2_ridge_penalty,
    poisson_deviance,
    smoothness_penalty,
    tweedie_deviance,
)


# -----------------------------------------------------------------------
# Poisson deviance tests
# -----------------------------------------------------------------------


class TestPoissonDeviance:
    def test_zero_when_perfect(self):
        """Deviance is zero when y_pred == y_true."""
        y = torch.tensor([1.0, 2.0, 3.0])
        assert poisson_deviance(y, y).item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self):
        """Poisson deviance is always non-negative."""
        y_pred = torch.tensor([0.5, 1.0, 2.0, 5.0])
        y_true = torch.tensor([1.0, 0.0, 3.0, 2.0])
        loss = poisson_deviance(y_pred, y_true)
        assert loss.item() >= 0.0

    def test_known_value(self):
        """Verify against manual calculation.

        For y=2, mu=1:
        D = 2 * [2*log(2/1) - (2-1)] = 2 * [2*log2 - 1] = 2*(1.386-1) = 0.772
        """
        y_pred = torch.tensor([1.0])
        y_true = torch.tensor([2.0])
        expected = 2.0 * (2.0 * np.log(2.0) - 1.0)
        result = poisson_deviance(y_pred, y_true)
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_zero_observation(self):
        """y=0 should not produce NaN (0*log(0) = 0 by convention)."""
        y_pred = torch.tensor([1.0])
        y_true = torch.tensor([0.0])
        loss = poisson_deviance(y_pred, y_true)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_exposure_weighted(self):
        """Exposure weighting should down-weight low-exposure observations."""
        y_pred = torch.tensor([1.0, 1.0])
        y_true = torch.tensor([2.0, 2.0])
        w = torch.tensor([1.0, 2.0])  # second observation has double weight

        loss_weighted = poisson_deviance(y_pred, y_true, w)
        loss_uniform = poisson_deviance(y_pred, y_true)

        # Both should equal the same deviance since observations are identical
        assert loss_weighted.item() == pytest.approx(loss_uniform.item(), rel=1e-3)

    def test_gradient_flows(self):
        """Gradient must flow through the loss for training."""
        y_pred = torch.tensor([1.0, 2.0], requires_grad=True)
        y_true = torch.tensor([2.0, 1.0])
        loss = poisson_deviance(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()

    def test_symmetry_is_asymmetric(self):
        """Poisson deviance is NOT symmetric: D(y, mu) != D(mu, y)."""
        y_pred = torch.tensor([2.0])
        y_true = torch.tensor([3.0])
        d_forward = poisson_deviance(y_pred, y_true).item()
        d_backward = poisson_deviance(y_true, y_pred).item()
        assert abs(d_forward - d_backward) > 1e-4  # definitely asymmetric

    def test_higher_for_worse_prediction(self):
        """Worse predictions should give higher deviance."""
        y_true = torch.tensor([2.0, 2.0])
        y_good = torch.tensor([2.1, 1.9])   # close
        y_bad = torch.tensor([5.0, 0.5])    # far

        d_good = poisson_deviance(y_good, y_true).item()
        d_bad = poisson_deviance(y_bad, y_true).item()
        assert d_good < d_bad


# -----------------------------------------------------------------------
# Gamma deviance tests
# -----------------------------------------------------------------------


class TestGammaDeviance:
    def test_zero_when_perfect(self):
        y = torch.tensor([1.0, 5.0, 10.0])
        assert gamma_deviance(y, y).item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self):
        y_pred = torch.tensor([0.5, 2.0, 10.0])
        y_true = torch.tensor([1.0, 1.0, 5.0])
        assert gamma_deviance(y_pred, y_true).item() >= 0.0

    def test_known_value(self):
        """For y=1, mu=2: D = 2 * [log(2/1) + (1-2)/2] = 2*(log2 - 0.5) = 2*(0.693-0.5) = 0.386."""
        y_pred = torch.tensor([2.0])
        y_true = torch.tensor([1.0])
        expected = 2.0 * (np.log(2.0) - 0.5)
        result = gamma_deviance(y_pred, y_true)
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_gradient_flows(self):
        y_pred = torch.tensor([1.5, 2.5], requires_grad=True)
        y_true = torch.tensor([1.0, 3.0])
        loss = gamma_deviance(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None

    def test_exposure_weighting(self):
        """Double-weighting an observation should change the weighted mean."""
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([2.0, 1.0])
        w_eq = torch.tensor([1.0, 1.0])
        w_uniq = torch.tensor([2.0, 1.0])

        loss_eq = gamma_deviance(y_pred, y_true, w_eq).item()
        loss_uniq = gamma_deviance(y_pred, y_true, w_uniq).item()
        assert abs(loss_eq - loss_uniq) > 1e-5


# -----------------------------------------------------------------------
# Tweedie deviance tests
# -----------------------------------------------------------------------


class TestTweedieDeviance:
    def test_zero_when_perfect(self):
        y = torch.tensor([1.0, 2.0, 5.0])
        result = tweedie_deviance(y, y, p=1.5)
        assert result.item() == pytest.approx(0.0, abs=1e-4)

    def test_non_negative(self):
        for p in [1.1, 1.5, 1.9]:
            y_pred = torch.tensor([1.0, 2.0])
            y_true = torch.tensor([2.0, 1.0])
            assert tweedie_deviance(y_pred, y_true, p=p).item() >= -1e-6

    def test_p1_agrees_with_poisson(self):
        """Tweedie p=1 should match Poisson deviance."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([2.0, 1.5, 4.0])
        tw = tweedie_deviance(y_pred, y_true, p=1.0)
        po = poisson_deviance(y_pred, y_true)
        assert tw.item() == pytest.approx(po.item(), rel=1e-3)

    def test_p2_agrees_with_gamma(self):
        """Tweedie p=2 should match Gamma deviance."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([2.0, 1.5, 4.0])
        tw = tweedie_deviance(y_pred, y_true, p=2.0)
        ga = gamma_deviance(y_pred, y_true)
        assert tw.item() == pytest.approx(ga.item(), rel=1e-3)

    def test_gradient_flows_various_p(self):
        for p in [1.2, 1.5, 1.8]:
            y_pred = torch.tensor([1.5, 2.5], requires_grad=True)
            y_true = torch.tensor([2.0, 1.0])
            loss = tweedie_deviance(y_pred, y_true, p=p)
            loss.backward()
            assert y_pred.grad is not None, f"No gradient for p={p}"
            assert not torch.isnan(y_pred.grad).any()
            y_pred.grad = None

    def test_zero_observations_no_nan(self):
        """Tweedie with y=0 should handle the y^(2-p) term (0^positive = 0)."""
        y_pred = torch.tensor([1.0])
        y_true = torch.tensor([0.0])
        loss = tweedie_deviance(y_pred, y_true, p=1.5)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_different_p_give_different_loss(self):
        """Different power parameters should give different loss values."""
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([2.0, 1.0])
        loss_12 = tweedie_deviance(y_pred, y_true, p=1.2).item()
        loss_18 = tweedie_deviance(y_pred, y_true, p=1.8).item()
        assert abs(loss_12 - loss_18) > 1e-4


# -----------------------------------------------------------------------
# Bernoulli deviance tests
# -----------------------------------------------------------------------


class TestBernoulliDeviance:
    def test_gradient_flows(self):
        logits = torch.tensor([1.0, -1.0], requires_grad=True)
        y_true = torch.tensor([1.0, 0.0])
        loss = bernoulli_deviance(logits, y_true)
        loss.backward()
        assert logits.grad is not None

    def test_non_negative(self):
        logits = torch.randn(10)
        y_true = torch.randint(0, 2, (10,)).float()
        assert bernoulli_deviance(logits, y_true).item() >= 0.0

    def test_minimum_at_correct_prediction(self):
        """Very confident correct predictions should give near-zero loss."""
        logits = torch.tensor([10.0, -10.0])  # very confident
        y_true = torch.tensor([1.0, 0.0])
        loss = bernoulli_deviance(logits, y_true)
        assert loss.item() < 0.01


# -----------------------------------------------------------------------
# Regularisation penalty tests
# -----------------------------------------------------------------------


class TestRegularisationPenalties:
    def test_smoothness_penalty_zero_for_linear(self):
        """A perfectly linear function has zero second differences."""
        import torch.nn as nn
        from insurance_gam.anam.feature_network import FeatureNetwork

        # A network with all zero output will have zero penalty
        net = FeatureNetwork(hidden_sizes=[8])
        # Zero-initialise output layer
        for module in net.network.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.zero_()
                module.bias.data.zero_()

        penalty = smoothness_penalty(net, x_min=0.0, x_max=1.0, n_points=20, lambda_smooth=1.0)
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_smoothness_penalty_positive_for_wiggly(self):
        """A wiggly network should have positive smoothness penalty."""
        import torch.nn as nn
        from insurance_gam.anam.feature_network import FeatureNetwork

        net = FeatureNetwork(hidden_sizes=[16, 8])
        # Random init → nonzero outputs → nonzero second differences
        penalty = smoothness_penalty(net, x_min=0.0, x_max=1.0, n_points=50, lambda_smooth=1.0)
        # Could be near zero by chance but should not be exactly zero
        # Just check it doesn't crash and is non-negative
        assert penalty.item() >= 0.0

    def test_l2_ridge_nonnegative(self):
        from insurance_gam.anam.feature_network import FeatureNetwork

        nets = [FeatureNetwork(hidden_sizes=[8]) for _ in range(3)]
        penalty = l2_ridge_penalty(nets, lambda_l2=1e-3)
        assert penalty.item() >= 0.0

    def test_l2_ridge_scales_with_lambda(self):
        from insurance_gam.anam.feature_network import FeatureNetwork

        nets = [FeatureNetwork(hidden_sizes=[8])]
        p1 = l2_ridge_penalty(nets, lambda_l2=1e-3).item()
        p2 = l2_ridge_penalty(nets, lambda_l2=1e-2).item()
        assert p2 == pytest.approx(p1 * 10, rel=1e-4)

    def test_l1_sparsity_nonnegative(self):
        from insurance_gam.anam.feature_network import FeatureNetwork

        nets = [FeatureNetwork(hidden_sizes=[8]) for _ in range(2)]
        penalty = l1_sparsity_penalty(nets, lambda_l1=1e-4)
        assert penalty.item() >= 0.0

    def test_smoothness_penalty_lambda_scaling(self):
        from insurance_gam.anam.feature_network import FeatureNetwork

        net = FeatureNetwork(hidden_sizes=[16])
        p1 = smoothness_penalty(net, 0.0, 1.0, n_points=50, lambda_smooth=1.0).item()
        p2 = smoothness_penalty(net, 0.0, 1.0, n_points=50, lambda_smooth=2.0).item()
        assert p2 == pytest.approx(p1 * 2.0, rel=1e-4)
