"""
test_utils.py — Tests for utility functions.
"""

import numpy as np
import polars as pl
import pytest
import torch

from insurance_gam.anam.utils import (
    StandardScaler,
    compare_shapes_to_glm,
    compute_deviance_stat,
    select_interactions_correlation,
    select_interactions_residual,
    shapes_to_relativity_table,
)
from insurance_gam.anam.shapes import ShapeFunction


# -----------------------------------------------------------------------
# StandardScaler tests
# -----------------------------------------------------------------------


class TestStandardScaler:
    def test_fit_transform_zero_mean(self):
        X = np.random.default_rng(0).standard_normal((100, 3)).astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10

    def test_fit_transform_unit_std(self):
        X = np.random.default_rng(1).standard_normal((100, 3)).astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 1e-8

    def test_inverse_transform_recovers_original(self):
        X = np.random.default_rng(2).standard_normal((50, 4)).astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_scaled)
        assert np.allclose(X, X_back, atol=1e-8)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.zeros((5, 2)))

    def test_inverse_transform_single_column(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.fit(X)
        x_col = np.array([0.0, 1.0])  # normalised
        x_orig = scaler.inverse_transform_col(x_col, col_idx=0)
        expected = x_col * scaler.stds_[0] + scaler.means_[0]
        assert np.allclose(x_orig, expected)

    def test_constant_feature_no_div_zero(self):
        """Constant column should not cause division by zero."""
        X = np.ones((10, 2))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.isfinite(X_scaled).all()

    def test_feature_names_stored(self):
        X = np.zeros((10, 3))
        scaler = StandardScaler()
        scaler.fit(X, feature_names=["a", "b", "c"])
        assert scaler.feature_names_ == ["a", "b", "c"]


# -----------------------------------------------------------------------
# Interaction selection tests
# -----------------------------------------------------------------------


class TestSelectInteractionsCorrelation:
    def make_correlated_data(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = 0.8 * x1 + 0.2 * rng.standard_normal(n)  # highly correlated
        x3 = rng.standard_normal(n)  # independent
        return np.column_stack([x1, x2, x3])

    def test_returns_list(self):
        X = self.make_correlated_data()
        result = select_interactions_correlation(X, ["x1", "x2", "x3"], threshold=0.5)
        assert isinstance(result, list)

    def test_finds_correlated_pair(self):
        X = self.make_correlated_data()
        result = select_interactions_correlation(X, ["x1", "x2", "x3"], threshold=0.5)
        names = [(a, b) for a, b, _ in result]
        assert ("x1", "x2") in names

    def test_excludes_low_correlation_pairs(self):
        X = self.make_correlated_data()
        result = select_interactions_correlation(X, ["x1", "x2", "x3"], threshold=0.9)
        # x1-x3 and x2-x3 should not appear at threshold 0.9
        names = [(a, b) for a, b, _ in result]
        assert ("x1", "x3") not in names

    def test_top_k_limit(self):
        X = self.make_correlated_data()
        result = select_interactions_correlation(
            X, ["x1", "x2", "x3"], threshold=0.0, top_k=1
        )
        assert len(result) <= 1

    def test_correlation_values_in_result(self):
        X = self.make_correlated_data()
        result = select_interactions_correlation(X, ["x1", "x2", "x3"], threshold=0.0)
        for _, _, corr in result:
            assert -1.0 <= corr <= 1.0

    def test_sorted_by_abs_correlation_descending(self):
        X = self.make_correlated_data()
        result = select_interactions_correlation(X, ["x1", "x2", "x3"], threshold=0.0)
        corrs = [abs(c) for _, _, c in result]
        assert corrs == sorted(corrs, reverse=True)


class TestSelectInteractionsResidual:
    def test_returns_top_k(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        resid = rng.standard_normal(200)
        result = select_interactions_residual(X, resid, ["a", "b", "c", "d"], top_k=3)
        assert len(result) <= 3

    def test_returns_list_of_tuples(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        resid = rng.standard_normal(100)
        result = select_interactions_residual(X, resid, ["a", "b", "c"], top_k=2)
        assert isinstance(result, list)
        assert all(len(t) == 3 for t in result)


# -----------------------------------------------------------------------
# GLM comparison utilities tests
# -----------------------------------------------------------------------


def make_test_shapes():
    sf_cont = ShapeFunction(
        feature_name="driver_age",
        feature_type="continuous",
        x_values=np.linspace(18, 80, 50),
        f_values=np.linspace(-0.5, 0.5, 50),
    )
    sf_cat = ShapeFunction(
        feature_name="region",
        feature_type="categorical",
        x_values=np.array([0.0, 1.0, 2.0, 3.0]),
        f_values=np.array([0.0, 0.2, -0.1, 0.3]),
        category_labels={0: "A", 1: "B", 2: "C", 3: "D"},
    )
    return {"driver_age": sf_cont, "region": sf_cat}


class TestShapesToRelativityTable:
    def test_returns_polars_dataframe(self):
        shapes = make_test_shapes()
        df = shapes_to_relativity_table(shapes)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self):
        shapes = make_test_shapes()
        df = shapes_to_relativity_table(shapes)
        for col in ["feature", "level", "f_x", "relativity", "log_relativity"]:
            assert col in df.columns

    def test_all_features_present(self):
        shapes = make_test_shapes()
        df = shapes_to_relativity_table(shapes)
        features = set(df["feature"].to_list())
        assert "driver_age" in features
        assert "region" in features

    def test_relativities_positive(self):
        shapes = make_test_shapes()
        df = shapes_to_relativity_table(shapes)
        assert (df["relativity"].to_numpy() > 0).all()

    def test_subset_features(self):
        shapes = make_test_shapes()
        df = shapes_to_relativity_table(shapes, feature_names=["driver_age"])
        assert set(df["feature"].to_list()) == {"driver_age"}


class TestCompareShapesToGLM:
    def test_returns_polars_dataframe(self):
        shapes = make_test_shapes()
        glm = {"driver_age": {"18.0": -0.5, "80.0": 0.5}}
        df = compare_shapes_to_glm(shapes, glm)
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self):
        shapes = make_test_shapes()
        glm = {"driver_age": {"30.0": 0.0}}
        df = compare_shapes_to_glm(shapes, glm)
        for col in ["feature", "level", "anam_f", "glm_log_rel", "deviation"]:
            assert col in df.columns

    def test_empty_when_no_matching_features(self):
        shapes = make_test_shapes()
        glm = {"unknown_feature": {"0": 0.0}}
        df = compare_shapes_to_glm(shapes, glm)
        assert len(df) == 0


class TestComputeDevianceStat:
    def test_poisson_deviance(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        stat = compute_deviance_stat(y_true, y_pred, loss="poisson")
        assert stat == pytest.approx(0.0, abs=1e-4)

    def test_gamma_deviance(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        stat = compute_deviance_stat(y_true, y_pred, loss="gamma")
        assert stat == pytest.approx(0.0, abs=1e-4)

    def test_tweedie_deviance(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        stat = compute_deviance_stat(y_true, y_pred, loss="tweedie", tweedie_p=1.5)
        assert stat == pytest.approx(0.0, abs=1e-4)

    def test_with_exposure_weights(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        w = np.array([1.0, 2.0, 0.5])
        stat = compute_deviance_stat(y_true, y_pred, exposure=w, loss="poisson")
        assert np.isfinite(stat)
        assert stat > 0.0

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            compute_deviance_stat(np.ones(3), np.ones(3), loss="unknown")
