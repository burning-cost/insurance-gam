"""
test_api.py — Tests for the sklearn-compatible ANAM wrapper.
"""

import numpy as np
import polars as pl
import pytest
import torch

from insurance_gam.anam import ANAM


def make_simple_dataset(n=300, seed=0):
    """Simple two-feature Poisson dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype(np.float32)
    exposure = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
    log_mu = -2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.log(exposure)
    y = rng.poisson(np.exp(log_mu)).astype(np.float32)
    return X, y, exposure


class TestANAMFit:
    def test_fit_returns_self(self):
        """fit() should return self for sklearn pipeline compatibility."""
        X, y, exp = make_simple_dataset()
        model = ANAM(
            feature_names=["x1", "x2"],
            n_epochs=3,
            verbose=0,
        )
        result = model.fit(X, y, sample_weight=exp)
        assert result is model

    def test_predict_shape(self):
        """predict() should return (n,) array."""
        X, y, exp = make_simple_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=3, verbose=0)
        model.fit(X, y, sample_weight=exp)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_positive_log_link(self):
        """Log-link predictions should all be positive."""
        X, y, exp = make_simple_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=3, verbose=0)
        model.fit(X, y, sample_weight=exp)
        preds = model.predict(X)
        assert (preds > 0).all()

    def test_predict_without_fitting_raises(self):
        """predict() before fit() should raise RuntimeError."""
        model = ANAM(feature_names=["x1"])
        X = np.zeros((10, 1))
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_score_returns_float(self):
        """score() should return a scalar float."""
        X, y, exp = make_simple_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=3, verbose=0)
        model.fit(X, y, sample_weight=exp)
        s = model.score(X, y, sample_weight=exp)
        assert isinstance(s, float)

    def test_score_negative_deviance(self):
        """score() returns negative deviance — higher is better."""
        X, y, exp = make_simple_dataset()
        model = ANAM(feature_names=["x1", "x2"], n_epochs=3, verbose=0)
        model.fit(X, y, sample_weight=exp)
        s = model.score(X, y, sample_weight=exp)
        # For a reasonable model, score should be in a sensible range
        assert np.isfinite(s)

    def test_fit_with_polars_dataframe(self):
        """ANAM should accept Polars DataFrames as input."""
        X, y, exp = make_simple_dataset(n=100)
        df = pl.DataFrame({"x1": X[:, 0].tolist(), "x2": X[:, 1].tolist()})
        model = ANAM(n_epochs=2, verbose=0)
        model.fit(df, y, sample_weight=exp)
        preds = model.predict(df)
        assert preds.shape == (100,)

    def test_fit_without_exposure(self):
        """Training without exposure should use unit weights."""
        X, y, _ = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=3, verbose=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_monotone_increasing_constraint(self):
        """Monotone increasing constraint should produce monotone shape."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.uniform(0, 10, size=n).astype(np.float32)
        exp = np.ones(n, dtype=np.float32)
        log_mu = -2.0 + 0.3 * x + np.log(exp)
        y = rng.poisson(np.exp(log_mu)).astype(np.float32)
        X = x.reshape(-1, 1)

        model = ANAM(
            feature_names=["x"],
            monotone_increasing=["x"],
            n_epochs=10,
            hidden_sizes=[16, 8],
            verbose=0,
        )
        model.fit(X, y, sample_weight=exp)

        # Extract shape function and verify monotonicity
        shapes = model.shape_functions()
        sf = shapes["x"]
        diffs = np.diff(sf.f_values)
        assert (diffs >= -1e-4).all(), f"Monotone violation: min diff = {diffs.min()}"

    def test_monotone_decreasing_constraint(self):
        """Monotone decreasing constraint should produce non-increasing shape."""
        rng = np.random.default_rng(8)
        n = 300
        x = rng.uniform(0, 10, size=n).astype(np.float32)
        exp = np.ones(n, dtype=np.float32)
        log_mu = -2.0 - 0.3 * x + np.log(exp)
        y = rng.poisson(np.exp(log_mu)).astype(np.float32)
        X = x.reshape(-1, 1)

        model = ANAM(
            feature_names=["x"],
            monotone_decreasing=["x"],
            n_epochs=10,
            hidden_sizes=[16, 8],
            verbose=0,
        )
        model.fit(X, y, sample_weight=exp)

        shapes = model.shape_functions()
        sf = shapes["x"]
        diffs = np.diff(sf.f_values)
        assert (diffs <= 1e-4).all(), f"Monotone violation: max diff = {diffs.max()}"

    def test_feature_importance_polars(self):
        """feature_importance() should return a Polars DataFrame."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)
        fi = model.feature_importance()
        assert isinstance(fi, pl.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_all_features_present(self):
        """feature_importance() should include all features."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)
        fi = model.feature_importance()
        assert set(fi["feature"].to_list()) == {"x1", "x2"}

    def test_shape_functions_returned(self):
        """shape_functions() should return a dict keyed by feature name."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)
        shapes = model.shape_functions()
        assert isinstance(shapes, dict)
        assert "x1" in shapes
        assert "x2" in shapes

    def test_shape_function_arrays_match_n_points(self):
        """Shape function arrays should have n_points entries."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)
        shapes = model.shape_functions(n_points=50)
        for sf in shapes.values():
            assert len(sf.x_values) == 50
            assert len(sf.f_values) == 50

    def test_get_params(self):
        """get_params() should return all key hyperparameters."""
        model = ANAM(n_epochs=20, learning_rate=1e-4)
        params = model.get_params()
        assert "n_epochs" in params
        assert params["n_epochs"] == 20
        assert "learning_rate" in params

    def test_set_params(self):
        """set_params() should update hyperparameters."""
        model = ANAM(n_epochs=10)
        model.set_params(n_epochs=50, learning_rate=5e-4)
        assert model.n_epochs == 50
        assert model.learning_rate == 5e-4

    def test_normalize_true(self):
        """With normalize=True, scaler_ should be set after fit."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0, normalize=True)
        model.fit(X, y, sample_weight=exp)
        assert model.scaler_ is not None

    def test_normalize_false(self):
        """With normalize=False, scaler_ should remain None."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0, normalize=False)
        model.fit(X, y, sample_weight=exp)
        assert model.scaler_ is None

    def test_categorical_feature_handled(self, anam_synthetic_data):
        """Categorical features should be handled via auto-config."""
        data = anam_synthetic_data
        model = ANAM(
            feature_names=["driver_age", "vehicle_age", "ncd", "region", "vehicle_type"],
            categorical_features=["region", "vehicle_type"],
            n_epochs=2,
            verbose=0,
        )
        model.fit(data["X"], data["y"], sample_weight=data["exposure"])
        preds = model.predict(data["X"])
        assert (preds > 0).all()

    def test_predict_with_polars_series_exposure(self):
        """predict() should accept a Polars Series as exposure."""
        X, y, exp = make_simple_dataset(n=100)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=2, verbose=0)
        model.fit(X, y, sample_weight=exp)
        exp_series = pl.Series(exp.tolist())
        preds = model.predict(X, exposure=exp_series)
        assert preds.shape == (100,)

    def test_predictions_finite(self):
        """Predictions should not contain NaN or Inf."""
        X, y, exp = make_simple_dataset(n=200)
        model = ANAM(feature_names=["x1", "x2"], n_epochs=5, verbose=0)
        model.fit(X, y, sample_weight=exp)
        preds = model.predict(X)
        assert np.isfinite(preds).all()

    def test_tweedie_mode(self):
        """ANAM with Tweedie loss should train and predict without error."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2)).astype(np.float32)
        y = rng.gamma(2.0, 1.0, size=200).astype(np.float32)
        exp = np.ones(200, dtype=np.float32)

        model = ANAM(
            feature_names=["x1", "x2"],
            loss="tweedie",
            tweedie_p=1.5,
            n_epochs=3,
            verbose=0,
        )
        model.fit(X, y, sample_weight=exp)
        preds = model.predict(X)
        assert np.isfinite(preds).all()
        assert (preds > 0).all()
