"""
Tests for BoulevardEBM and BoulevardInference.

Test strategy:
  - No mocking — Boulevard has no optional dependencies beyond numpy, scipy,
    sklearn.  interpretML is deliberately NOT required.
  - Run on synthetic data only; nothing touches the network.
  - CI coverage test uses a generous 90% empirical coverage threshold for a
    nominal 95% CI: asymptotic theory applies in large samples and the
    algorithm has a finite-sample bias from the moving-average warm-up.
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl

from insurance_gam.ebm import BoulevardEBM, BoulevardInference


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_sine_data(n: int = 500, noise_std: float = 0.2, seed: int = 0) -> tuple:
    """y = sin(2*pi*x) + N(0, noise_std^2), x ~ Uniform(0,1)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, noise_std, size=n)
    X = x.reshape(-1, 1)
    return X, y


def _make_additive_data(n: int = 600, noise_std: float = 0.3, seed: int = 1) -> tuple:
    """y = sin(2*pi*x1) + 0.5*x2^2 + N(0, noise_std^2)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 ** 2 + rng.normal(0.0, noise_std, size=n)
    X = np.column_stack([x1, x2])
    return X, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_sine_model():
    X, y = _make_sine_data(n=600)
    model = BoulevardEBM(n_estimators=500, lambda_=1.0, max_bins=64, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture(scope="module")
def fitted_additive_model():
    X, y = _make_additive_data(n=600)
    df = pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
    model = BoulevardEBM(n_estimators=500, lambda_=1.0, max_bins=64, random_state=42)
    model.fit(df, y)
    return model, df, y


@pytest.fixture(scope="module")
def fitted_sine_inference(fitted_sine_model):
    model, X, y = fitted_sine_model
    return BoulevardInference(model), model, X, y


# ---------------------------------------------------------------------------
# Test 1: Shape recovery — sine function
# ---------------------------------------------------------------------------

class TestShapeRecovery:
    def test_shape_scores_not_all_zero(self, fitted_sine_model):
        model, X, y = fitted_sine_model
        edges, scores = model.shape_function(0)
        assert np.std(scores) > 0.05, "Shape scores should vary, not be flat"

    def test_shape_tracks_sine(self, fitted_sine_model):
        """Shape function should positively correlate with sin(2*pi*x)."""
        model, X, y = fitted_sine_model
        edges, scores = model.shape_function(0)
        n_bins = len(scores)
        # bin midpoints (ignore +inf final edge)
        mids = np.array([
            (edges[j] + edges[j + 1]) / 2.0 if not np.isinf(edges[j + 1]) else edges[j]
            for j in range(n_bins)
        ])
        true_shape = np.sin(2 * np.pi * mids)
        corr = float(np.corrcoef(true_shape, scores)[0, 1])
        assert corr > 0.85, f"Pearson correlation with true sine = {corr:.3f}, expected > 0.85"

    def test_predict_mean_near_intercept(self, fitted_sine_model):
        """Since shape scores are centred, predict(X_train) mean ≈ intercept."""
        model, X, y = fitted_sine_model
        preds = model.predict(X)
        assert abs(np.mean(preds) - model.intercept_) < 0.1


# ---------------------------------------------------------------------------
# Test 2: CI coverage
# ---------------------------------------------------------------------------

class TestCICoverage:
    def test_ci_coverage_90_percent(self):
        """
        Empirical coverage of 95% CIs should be ≥ 90% at bin midpoints.

        Uses a generous threshold because:
        - Asymptotic CIs are exact only as n → ∞
        - The Boulevard moving-average has a finite-sample warm-up bias
        - We only run 50 Monte Carlo repetitions to keep the test fast
        """
        n_mc = 50
        n_obs = 400
        noise_std = 0.2
        alpha = 0.05
        n_bins = 32
        covered_bins = []

        for seed in range(n_mc):
            X, y = _make_sine_data(n=n_obs, noise_std=noise_std, seed=seed)
            model = BoulevardEBM(
                n_estimators=300, lambda_=1.0, max_bins=n_bins, random_state=seed
            )
            model.fit(X, y)
            inf = BoulevardInference(model)
            ci_df = inf.shape_ci(0, alpha=alpha)

            lower = ci_df["lower"].to_numpy()
            upper = ci_df["upper"].to_numpy()
            edges = model.bins_[0]
            n_b = len(lower)
            mids = np.array([
                (edges[j] + edges[j + 1]) / 2.0 if not np.isinf(edges[j + 1]) else edges[j]
                for j in range(n_b)
            ])
            true_vals = np.sin(2 * np.pi * mids)

            # Coverage: true value inside CI (ignoring the intercept shift —
            # we compare centred shapes by aligning true_vals to zero mean)
            true_centred = true_vals - np.mean(true_vals)
            covered = (lower <= true_centred) & (true_centred <= upper)
            covered_bins.append(float(np.mean(covered)))

        mean_coverage = float(np.mean(covered_bins))
        assert mean_coverage >= 0.90, (
            f"Mean bin coverage = {mean_coverage:.3f}, expected ≥ 0.90 "
            f"for nominal 95% CIs"
        )


# ---------------------------------------------------------------------------
# Test 3: predict output shape
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_shape_matches_input(self, fitted_sine_model):
        model, X, y = fitted_sine_model
        preds = model.predict(X)
        assert preds.shape == (len(y),), f"Expected ({len(y)},), got {preds.shape}"

    def test_predict_on_subset(self, fitted_sine_model):
        model, X, y = fitted_sine_model
        subset = X[:10]
        preds = model.predict(subset)
        assert preds.shape == (10,)

    def test_predict_is_float(self, fitted_sine_model):
        model, X, y = fitted_sine_model
        preds = model.predict(X[:5])
        assert preds.dtype.kind == "f"

    def test_predict_unfitted_raises(self):
        model = BoulevardEBM()
        X = np.random.default_rng(0).uniform(size=(10, 1))
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(X)


# ---------------------------------------------------------------------------
# Test 4: BoulevardInference.shape_ci columns
# ---------------------------------------------------------------------------

class TestShapeCI:
    def test_shape_ci_columns(self, fitted_sine_inference):
        inf, model, X, y = fitted_sine_inference
        ci_df = inf.shape_ci(0)
        expected_cols = {"bin_label", "score", "lower", "upper", "se"}
        assert set(ci_df.columns) == expected_cols

    def test_shape_ci_length_matches_bins(self, fitted_sine_inference):
        inf, model, X, y = fitted_sine_inference
        ci_df = inf.shape_ci(0)
        n_bins = len(model.shape_scores_[0])
        assert len(ci_df) == n_bins

    def test_shape_ci_lower_le_score_le_upper(self, fitted_sine_inference):
        inf, model, X, y = fitted_sine_inference
        ci_df = inf.shape_ci(0)
        lower = ci_df["lower"].to_numpy()
        upper = ci_df["upper"].to_numpy()
        scores = ci_df["score"].to_numpy()
        assert np.all(lower <= scores + 1e-10), "lower should be ≤ score"
        assert np.all(scores <= upper + 1e-10), "score should be ≤ upper"

    def test_shape_ci_wider_at_alpha_005_vs_020(self, fitted_sine_inference):
        inf, model, X, y = fitted_sine_inference
        ci_95 = inf.shape_ci(0, alpha=0.05)
        ci_80 = inf.shape_ci(0, alpha=0.20)
        width_95 = (ci_95["upper"] - ci_95["lower"]).to_numpy()
        width_80 = (ci_80["upper"] - ci_80["lower"]).to_numpy()
        assert np.all(width_95 >= width_80 - 1e-10), "95% CI should be wider than 80% CI"

    def test_shape_ci_se_positive(self, fitted_sine_inference):
        inf, model, X, y = fitted_sine_inference
        ci_df = inf.shape_ci(0)
        se = ci_df["se"].to_numpy()
        assert np.all(se >= 0.0), "SE should be non-negative"

    def test_shape_ci_by_name(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        ci_df = inf.shape_ci("x1")
        assert "bin_label" in ci_df.columns


# ---------------------------------------------------------------------------
# Test 5: significance_table — one row per feature
# ---------------------------------------------------------------------------

class TestSignificanceTable:
    def test_significance_table_row_count(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        tbl = inf.significance_table()
        assert len(tbl) == 2, f"Expected 2 rows (one per feature), got {len(tbl)}"

    def test_significance_table_columns(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        tbl = inf.significance_table()
        expected = {"feature", "max_absolute_score", "ci_excludes_zero", "max_se"}
        assert set(tbl.columns) == expected

    def test_significance_table_feature_names(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        tbl = inf.significance_table()
        assert list(tbl["feature"]) == ["x1", "x2"]

    def test_significance_table_max_abs_nonneg(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        tbl = inf.significance_table()
        assert (tbl["max_absolute_score"] >= 0.0).all()

    def test_significance_table_single_feature(self, fitted_sine_model):
        model, X, y = fitted_sine_model
        inf = BoulevardInference(model)
        tbl = inf.significance_table()
        assert len(tbl) == 1


# ---------------------------------------------------------------------------
# Test 6: plot_shape doesn't error
# ---------------------------------------------------------------------------

class TestPlotShape:
    def test_plot_shape_returns_axes(self, fitted_sine_inference):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend; safe in CI
        import matplotlib.pyplot as plt

        inf, model, X, y = fitted_sine_inference
        ax = inf.plot_shape(0)
        assert ax is not None
        plt.close("all")

    def test_plot_shape_no_density(self, fitted_sine_inference):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        inf, model, X, y = fitted_sine_inference
        ax = inf.plot_shape(0, show_data_density=False)
        assert ax is not None
        plt.close("all")

    def test_plot_shape_by_name(self, fitted_additive_model):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        ax = inf.plot_shape("x2")
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Test 7: polars DataFrame input
# ---------------------------------------------------------------------------

class TestPolarsInput:
    def test_fit_with_polars_dataframe(self):
        X_np, y = _make_sine_data(n=200)
        df = pl.DataFrame({"feature_a": X_np[:, 0]})
        model = BoulevardEBM(n_estimators=100, max_bins=32, random_state=0)
        model.fit(df, y)
        assert model.feature_names_ == ["feature_a"]
        assert model._fitted

    def test_predict_with_polars_dataframe(self):
        X_np, y = _make_sine_data(n=200)
        df = pl.DataFrame({"feature_a": X_np[:, 0]})
        model = BoulevardEBM(n_estimators=100, max_bins=32, random_state=0)
        model.fit(df, y)
        preds = model.predict(df)
        assert preds.shape == (200,)

    def test_shape_function_by_name_polars(self):
        X_np, y = _make_sine_data(n=200)
        df = pl.DataFrame({"temperature": X_np[:, 0]})
        model = BoulevardEBM(n_estimators=100, max_bins=32, random_state=0)
        model.fit(df, y)
        edges, scores = model.shape_function("temperature")
        assert len(scores) == len(edges) - 1

    def test_inference_shape_ci_returns_polars_df(self):
        X_np, y = _make_sine_data(n=200)
        df = pl.DataFrame({"temperature": X_np[:, 0]})
        model = BoulevardEBM(n_estimators=100, max_bins=32, random_state=0)
        model.fit(df, y)
        inf = BoulevardInference(model)
        ci_df = inf.shape_ci("temperature")
        assert isinstance(ci_df, pl.DataFrame)


# ---------------------------------------------------------------------------
# Test 8: multi-feature additive model
# ---------------------------------------------------------------------------

class TestMultiFeature:
    def test_fit_two_features(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        assert len(model.feature_names_) == 2
        assert len(model.shape_scores_) == 2
        assert len(model.bins_) == 2
        assert len(model.bin_counts_) == 2

    def test_each_feature_has_shape_variation(self, fitted_additive_model):
        """Both features should have non-trivial shape functions."""
        model, df, y = fitted_additive_model
        for k, fname in enumerate(model.feature_names_):
            std_k = np.std(model.shape_scores_[k])
            assert std_k > 0.01, (
                f"Feature {fname} shape scores have std={std_k:.4f}, expected > 0.01"
            )

    def test_predict_shape(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        preds = model.predict(df)
        assert preds.shape == (len(y),)

    def test_inference_on_multi_feature(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        for fname in ["x1", "x2"]:
            ci_df = inf.shape_ci(fname)
            assert len(ci_df) > 0
            assert "se" in ci_df.columns

    def test_additive_model_r2(self, fitted_additive_model):
        """Fitted model should explain a reasonable fraction of variance."""
        model, df, y = fitted_additive_model
        preds = model.predict(df)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.3, f"R² = {r2:.3f}, expected > 0.30"

    def test_prediction_interval_columns(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        pi_df = inf.prediction_interval(df)
        assert set(pi_df.columns) == {"lower", "prediction", "upper"}
        assert len(pi_df) == len(y)

    def test_prediction_interval_ordering(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        inf = BoulevardInference(model)
        pi_df = inf.prediction_interval(df)
        lower = pi_df["lower"].to_numpy()
        pred = pi_df["prediction"].to_numpy()
        upper = pi_df["upper"].to_numpy()
        assert np.all(lower <= pred + 1e-10)
        assert np.all(pred <= upper + 1e-10)


# ---------------------------------------------------------------------------
# Test 9: BoulevardInference on unfitted model raises
# ---------------------------------------------------------------------------

class TestUnfittedInference:
    def test_inference_unfitted_raises(self):
        model = BoulevardEBM()
        with pytest.raises(RuntimeError, match="fitted"):
            BoulevardInference(model)


# ---------------------------------------------------------------------------
# Test 10: feature_index lookup
# ---------------------------------------------------------------------------

class TestFeatureIndex:
    def test_integer_index_valid(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        edges, scores = model.shape_function(0)
        assert len(scores) > 0

    def test_string_name_invalid_raises(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        with pytest.raises(ValueError, match="not found"):
            model.shape_function("nonexistent_feature")

    def test_string_name_valid(self, fitted_additive_model):
        model, df, y = fitted_additive_model
        edges, scores = model.shape_function("x2")
        assert len(scores) > 0
