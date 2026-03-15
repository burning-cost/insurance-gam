"""
test_p0_bugs.py — Regression tests for P0 bugs fixed in the audit.

One test class per bug, named after the bug ID so failures are
immediately identifiable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from insurance_gam.anam import ANAM
from insurance_gam.anam.feature_network import ExUActivation, FeatureNetwork
from sklearn.base import clone


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_continuous_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype(np.float32)
    exposure = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
    log_mu = -2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.log(exposure)
    y = rng.poisson(np.exp(log_mu)).astype(np.float32)
    return X, y, exposure


# ---------------------------------------------------------------------------
# P0-1: shape_functions() cache ignores n_points on subsequent calls
# ---------------------------------------------------------------------------


class TestP01ShapeCacheNPoints:
    """shape_functions() must respect n_points on every call, not just the first."""

    def test_cache_returns_correct_length_for_first_call(self):
        X, y, exp = _make_continuous_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)

        shapes = model.shape_functions(n_points=100)
        for sf in shapes.values():
            assert len(sf.x_values) == 100, (
                f"Expected 100 points, got {len(sf.x_values)}"
            )

    def test_cache_invalidated_on_different_n_points(self):
        """Calling with n_points=200 after n_points=50 must return 200-point result."""
        X, y, exp = _make_continuous_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)

        # Warm the cache with 50 points.
        shapes_50 = model.shape_functions(n_points=50)
        for sf in shapes_50.values():
            assert len(sf.x_values) == 50

        # Now request 200 points — must not return stale 50-point result.
        shapes_200 = model.shape_functions(n_points=200)
        for sf in shapes_200.values():
            assert len(sf.x_values) == 200, (
                f"Cache was not invalidated: expected 200 points, got {len(sf.x_values)}"
            )

    def test_cache_reused_for_same_n_points(self):
        """Two calls with the same n_points should return the identical dict object."""
        X, y, exp = _make_continuous_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)

        first = model.shape_functions(n_points=75)
        second = model.shape_functions(n_points=75)
        assert first is second, "Cache was not reused for the same n_points"

    def test_cache_invalidated_going_back_to_original_n_points(self):
        """50 -> 200 -> 50 must each give the correct point count."""
        X, y, exp = _make_continuous_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)

        model.shape_functions(n_points=50)
        model.shape_functions(n_points=200)
        shapes = model.shape_functions(n_points=50)
        for sf in shapes.values():
            assert len(sf.x_values) == 50


# ---------------------------------------------------------------------------
# P0-2: Categorical encoding not validated for zero-indexed consecutive ints
# ---------------------------------------------------------------------------


class TestP02CategoricalEncoding:
    """Categorical features must be validated and remapped if not zero-indexed."""

    def _make_cat_dataset(self, category_values: list[int], n: int = 300, seed: int = 1):
        """Dataset with one continuous + one categorical feature.

        category_values is the set of integer codes that appear in the data.
        """
        rng = np.random.default_rng(seed)
        x_cont = rng.standard_normal(n).astype(np.float32)
        x_cat = rng.choice(category_values, size=n).astype(np.float32)
        X = np.column_stack([x_cont, x_cat])
        exposure = np.ones(n, dtype=np.float32)
        # Simple log-linear truth
        cat_effect = {v: i * 0.1 for i, v in enumerate(sorted(category_values))}
        log_mu = -2.0 + 0.3 * x_cont + np.array([cat_effect[int(c)] for c in x_cat])
        y = rng.poisson(np.exp(log_mu)).astype(np.float32)
        return X, y, exposure

    def test_zero_indexed_categorical_no_warning(self):
        """Zero-indexed categories (0, 1, 2, 3) must fit without warnings."""
        X, y, exp = self._make_cat_dataset([0, 1, 2, 3])
        model = ANAM(
            feature_names=["cont", "region"],
            categorical_features=["region"],
            n_epochs=2,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.fit(X, y, sample_weight=exp)

        preds = model.predict(X)
        assert (preds > 0).all()

    def test_one_indexed_categorical_warns_and_works(self):
        """Categories starting at 1 must trigger a UserWarning and still fit correctly."""
        X, y, exp = self._make_cat_dataset([1, 2, 3, 4])
        model = ANAM(
            feature_names=["cont", "region"],
            categorical_features=["region"],
            n_epochs=2,
            verbose=0,
        )
        with pytest.warns(UserWarning, match="zero-indexed"):
            model.fit(X, y, sample_weight=exp)

        preds = model.predict(X)
        assert (preds > 0).all()
        assert np.isfinite(preds).all()

    def test_non_consecutive_categorical_warns_and_works(self):
        """Non-consecutive codes (e.g. UK region codes 1, 3, 7) must warn and fit."""
        X, y, exp = self._make_cat_dataset([1, 3, 7])
        model = ANAM(
            feature_names=["cont", "region"],
            categorical_features=["region"],
            n_epochs=2,
            verbose=0,
        )
        with pytest.warns(UserWarning, match="zero-indexed"):
            model.fit(X, y, sample_weight=exp)

        preds = model.predict(X)
        assert (preds > 0).all()

    def test_remap_applied_at_predict_time(self):
        """The same category remap used at fit must be applied at predict time."""
        # Categories 10, 20, 30 — large gap, definitely non-consecutive.
        X, y, exp = self._make_cat_dataset([10, 20, 30], n=400)
        model = ANAM(
            feature_names=["cont", "region"],
            categorical_features=["region"],
            n_epochs=2,
            verbose=0,
        )
        with pytest.warns(UserWarning):
            model.fit(X, y, sample_weight=exp)

        # Predict on the same X (with original codes 10, 20, 30).
        # If remap is not applied at predict time this would throw an
        # embedding index-out-of-range error.
        preds = model.predict(X)
        assert preds.shape == (400,)
        assert np.isfinite(preds).all()

    def test_zero_indexed_n_categories_correct(self):
        """With codes 0..3, the model must learn 4 categories (not 5)."""
        X, y, exp = self._make_cat_dataset([0, 1, 2, 3])
        model = ANAM(
            feature_names=["cont", "region"],
            categorical_features=["region"],
            n_epochs=2,
            verbose=0,
        )
        model.fit(X, y, sample_weight=exp)
        # Access the categorical subnetwork via model.model_.feature_nets
        cat_net = model.model_.feature_nets["region"]
        assert cat_net.n_categories == 4


# ---------------------------------------------------------------------------
# P0-3: ExU activation is a documented stub
# ---------------------------------------------------------------------------


class TestP03ExUActivation:
    """ExUActivation must actually be used when activation='exu'."""

    def test_exu_activation_module_present_in_network(self):
        """With activation='exu', the network layers must contain ExUActivation."""
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="exu")
        exu_layers = [m for m in net.network.modules() if isinstance(m, ExUActivation)]
        assert len(exu_layers) > 0, (
            "No ExUActivation found in network — ExU was not wired up."
        )

    def test_exu_no_extra_relu_in_hidden_layers(self):
        """With activation='exu', nn.ReLU should not appear as a standalone hidden layer.

        ExU fuses the linear transform into the activation, so no separate
        nn.Linear or nn.ReLU should appear in the hidden portion.
        """
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="exu")
        layers = list(net.network.children())
        relu_layers = [l for l in layers if isinstance(l, torch.nn.ReLU)]
        assert len(relu_layers) == 0, (
            f"Found standalone nn.ReLU layers in an ExU network: {relu_layers}"
        )

    def test_exu_forward_produces_correct_shape(self):
        """ExU network must produce (batch, 1) output for scalar inputs."""
        torch.manual_seed(0)
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="exu")
        x = torch.randn(32)
        out = net(x)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"

    def test_exu_output_finite(self):
        """ExU network must produce finite outputs for standard inputs."""
        torch.manual_seed(42)
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="exu")
        x = torch.linspace(-3.0, 3.0, 100)
        out = net(x)
        assert torch.isfinite(out).all(), "ExU network produced non-finite outputs"

    def test_exu_gradients_flow(self):
        """Gradients must reach the input through the ExU activation."""
        torch.manual_seed(1)
        net = FeatureNetwork(hidden_sizes=[16], activation="exu")
        x = torch.randn(10, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None, "No gradient reached the input through ExU"

    def test_relu_network_has_no_exu_layers(self):
        """Sanity check: relu networks must not accidentally contain ExUActivation."""
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="relu")
        exu_layers = [m for m in net.network.modules() if isinstance(m, ExUActivation)]
        assert len(exu_layers) == 0

    def test_exu_network_trains_via_optimizer(self):
        """End-to-end: FeatureNetwork with exu activation must accept gradient steps."""
        torch.manual_seed(5)
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="exu")
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        x = torch.randn(32)
        target = torch.randn(32)

        net.train()
        for _ in range(3):
            optimizer.zero_grad()
            out = net(x).squeeze(-1)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        assert torch.isfinite(torch.tensor(loss.item())), "Training loss is not finite"


# ---------------------------------------------------------------------------
# P0-4: get_params() omits all structural parameters
# ---------------------------------------------------------------------------


class TestP04GetParamsComplete:
    """get_params() must include all constructor parameters for sklearn clone()."""

    STRUCTURAL_PARAMS = [
        "feature_configs",
        "feature_names",
        "categorical_features",
        "monotone_increasing",
        "monotone_decreasing",
        "interaction_pairs",
    ]

    def test_get_params_includes_structural_params(self):
        """All structural parameters must appear in get_params() output."""
        model = ANAM(
            feature_names=["x1", "x2"],
            categorical_features=["x2"],
            monotone_increasing=["x1"],
        )
        params = model.get_params()
        for key in self.STRUCTURAL_PARAMS:
            assert key in params, f"'{key}' missing from get_params()"

    def test_get_params_values_match_constructor_args(self):
        """get_params() values must equal the arguments passed to __init__."""
        from insurance_gam.anam.model import FeatureConfig

        feat_configs = [
            FeatureConfig(name="age", feature_type="continuous"),
            FeatureConfig(name="region", feature_type="categorical", n_categories=4),
        ]
        model = ANAM(
            feature_configs=feat_configs,
            feature_names=["age", "region"],
            categorical_features=["region"],
            monotone_increasing=["age"],
            monotone_decreasing=[],
            interaction_pairs=[("age", "region")],
            n_epochs=50,
            learning_rate=5e-4,
        )
        params = model.get_params()
        assert params["feature_configs"] is feat_configs
        assert params["feature_names"] == ["age", "region"]
        assert params["categorical_features"] == ["region"]
        assert params["monotone_increasing"] == ["age"]
        assert params["monotone_decreasing"] == []
        assert params["interaction_pairs"] == [("age", "region")]
        assert params["n_epochs"] == 50
        assert params["learning_rate"] == 5e-4

    def test_clone_produces_unfitted_model_with_same_params(self):
        """sklearn.base.clone() must work and produce an unfitted model."""
        model = ANAM(
            feature_names=["x1", "x2"],
            monotone_increasing=["x1"],
            n_epochs=5,
            learning_rate=2e-3,
            hidden_sizes=[32, 16],
        )
        cloned = clone(model)

        # Cloned model must not be fitted.
        assert cloned.model_ is None

        # Structural params must be preserved.
        assert cloned.feature_names == ["x1", "x2"]
        assert cloned.monotone_increasing == ["x1"]
        assert cloned.n_epochs == 5
        assert cloned.learning_rate == 2e-3
        assert cloned.hidden_sizes == [32, 16]

    def test_clone_can_fit(self):
        """A cloned ANAM must be fittable and produce valid predictions."""
        X, y, exp = _make_continuous_dataset()
        original = ANAM(
            feature_names=["x1", "x2"],
            n_epochs=3,
            verbose=0,
        )
        original.fit(X, y, sample_weight=exp)

        cloned = clone(original)
        assert cloned.model_ is None, "clone() returned a fitted model"

        cloned.fit(X, y, sample_weight=exp)
        preds = cloned.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()
        assert (preds > 0).all()

    def test_clone_preserves_feature_configs(self):
        """clone() must carry feature_configs through to the new instance."""
        from insurance_gam.anam.model import FeatureConfig

        configs = [
            FeatureConfig(name="a", feature_type="continuous", monotonicity="increasing"),
        ]
        model = ANAM(feature_configs=configs)
        cloned = clone(model)
        assert cloned.feature_configs is not None
        assert len(cloned.feature_configs) == 1
        assert cloned.feature_configs[0].name == "a"


# ---------------------------------------------------------------------------
# Regression tests for the NEW P0/P1 bugs (second audit batch)
# ---------------------------------------------------------------------------


class TestP03ExUNegativeWeightsRegression:
    """ExU activation must produce non-zero output when weights are negative.

    The original bug: the forward pass used self.weights directly instead of
    torch.exp(self.weights). For negative weights, relu(x * w) would collapse to
    relu((negative) * (positive_input)) = 0 for all positive inputs.
    With exp(w): even w=-10 gives exp(-10) > 0, so relu(x * exp(w)) can be nonzero.
    """

    def test_exu_output_nonzero_with_negative_weights(self):
        """ExU must produce non-zero output when weights are initialised negative."""
        import torch.nn as nn
        net = FeatureNetwork(hidden_sizes=[16, 8], activation="exu")

        # Force all ExU weights to be strongly negative
        with torch.no_grad():
            for module in net.network.modules():
                if isinstance(module, ExUActivation):
                    module.weights.fill_(-5.0)  # exp(-5) = 0.0067 > 0
                    module.biases.fill_(0.0)

        x = torch.linspace(0.1, 1.0, 20)
        out = net(x)
        # With self.weights directly: relu(x * (-5)) = 0 for all positive x
        # With torch.exp(self.weights): relu(x * exp(-5)) can be nonzero -> nonzero output
        # The output layer applies a Linear on top, so check at the ExU layer directly
        exu_layer = [m for m in net.network.modules() if isinstance(m, ExUActivation)][0]
        with torch.no_grad():
            x_in = torch.linspace(0.1, 1.0, 20).unsqueeze(-1)  # (20, 1) -> in_features=1
            exu_out = exu_layer(x_in)
        # exp(-5.0) > 0, so shifted * exp(w) > 0 for x > bias(=0), relu keeps it
        assert exu_out.abs().sum().item() > 0, (
            "ExU output is all-zero with negative weights. "
            "This indicates torch.exp(self.weights) is not being applied."
        )

    def test_exu_uses_exp_not_raw_weights(self):
        """Verify the exp transformation: output must scale with exp(w) not w."""
        # For ExU: out = relu(exp(w) * (x - b))
        # If w=0: out = relu(1.0 * (x - b)) (exp(0)=1)
        # If w=2: out = relu(exp(2) * (x - b)) ~= relu(7.4 * (x - b)) (much larger)
        # If code uses w directly: out at w=2 would be relu(2 * (x - b)) (only 2x)
        exu = ExUActivation(in_features=1, out_features=4)
        with torch.no_grad():
            exu.weights.fill_(0.0)
            exu.biases.fill_(0.0)
        x = torch.tensor([[1.0]])  # (1, 1)
        out_w0 = exu(x).detach().clone()

        with torch.no_grad():
            exu.weights.fill_(2.0)
        out_w2 = exu(x).detach().clone()

        # exp(2) / exp(0) = e^2 ~= 7.389
        # If using raw weights: ratio = 2 / 0 = undefined (w=0 gives relu(0)=0)
        # So test with w=1 and w=3 instead:
        with torch.no_grad():
            exu.weights.fill_(1.0)
        out_w1 = exu(x).detach()

        with torch.no_grad():
            exu.weights.fill_(3.0)
        out_w3 = exu(x).detach()

        if out_w1.abs().sum() > 1e-9:
            ratio = (out_w3.abs().sum() / out_w1.abs().sum()).item()
            # exp(3)/exp(1) = e^2 ~= 7.39
            # raw weights: 3/1 = 3
            assert ratio > 5.0, (
                f"ExU output ratio for w=3 vs w=1 is {ratio:.2f}, expected ~7.4 (exp scaling). "
                f"Got {ratio:.2f} which suggests raw weight multiplication, not exp."
            )
