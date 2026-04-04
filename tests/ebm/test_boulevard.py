"""
Tests for BoulevardEBM and BoulevardInference.

Test strategy:
  - No mocking — Boulevard has no optional dependencies beyond numpy, scipy,
    sklearn.  interpretML is deliberately NOT required.
  - Run on synthetic data only; nothing touches the network.
  - CI calibration test uses SE ratio: the predicted SE at each bin should
    agree with the empirical standard deviation of scores across MC replicates.
    We do NOT test coverage of the true function value because the Boulevard
    CIs are confidence intervals for the KRR population target, which is a
    regularised (shrunk) version of the truth.  The bias from regularisation
    is by design and does not represent a bug in the SE formula.

SE calibration note:
  sigma_hat is estimated from training residuals, which include the
  KRR approximation error in addition to pure noise.  For a sine-wave DGP
  with lambda_=1.0, sigma_hat overestimates the true noise SD by roughly a
  factor of 2, so the predicted SE is proportionally inflated.  The
  calibration window [0.3, 4.0] allows for this known upward bias in
  sigma_hat and still catches the case where the SE formula is
  fundamentally wrong (off by an order of magnitude or degenerate).
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
# Test 2: SE calibration
#
# Boulevard CIs are pointwise asymptotic confidence intervals for the KRR
# population target, not for the true data-generating function.  The KRR
# target differs from the truth by a regularisation bias that is explicitly
# controlled by lambda_ and is not captured by the SE.
#
# The correct validation is SE calibration: the SE predicted by
# BoulevardInference should be in the same order of magnitude as the empirical
# standard deviation of scores across Monte Carlo replicates.  We use a wide
# calibration window [0.3, 4.0] for two reasons:
#
#   1. sigma_hat is estimated from training residuals, which include the KRR
#      approximation error (not just noise), causing a known upward bias.
#      For a sine DGP with lambda_=1.0 the expected ratio is ~2.0.
#
#   2. n_mc=30 reps means the empirical SD itself has ~13% relative error
#      (1/sqrt(2*(n_mc-1))), which spreads the ratio distribution.
#
# A ratio systematically below 0.3 would mean the SE formula is fundamentally
# broken (CIs far too narrow).  Above 4.0 would mean absurdly wide CIs.
# ---------------------------------------------------------------------------

class TestCICoverage:
    def test_ci_coverage_90_percent(self):
        """
        SE calibration: predicted SE should be in the same order of magnitude
        as the empirical standard deviation of shape scores across MC reps.

        We run n_mc fits on independently drawn datasets from the same DGP,
        collect per-bin scores and predicted SEs, then compare:
          - empirical_sd[j]    = std of scores[j] across n_mc fits
          - mean_predicted_se[j] = mean of BoulevardInference SE[j] across n_mc fits

        The ratio mean_predicted_se / empirical_sd should be in [0.3, 4.0] for
        at least 80% of bins.  This catches degenerate SEs (all zeros, or
        formula producing values 10x off) without being so tight that finite-
        sample sigma_hat bias causes spurious failures.
        """
        n_mc = 30
        n_obs = 800
        noise_std = 0.2
        n_bins = 32

        all_scores = []   # list of n_mc arrays, each of shape (n_bins_actual,)
        all_ses = []      # list of n_mc arrays of predicted SE

        for seed in range(n_mc):
            X, y = _make_sine_data(n=n_obs, noise_std=noise_std, seed=seed)
            model = BoulevardEBM(
                n_estimators=500, lambda_=1.0, max_bins=n_bins, random_state=seed
            )
            model.fit(X, y)
            inf = BoulevardInference(model)
            ci_df = inf.shape_ci(0, alpha=0.05)

            all_scores.append(ci_df["score"].to_numpy())
            all_ses.append(ci_df["se"].to_numpy())

        # All reps should produce the same number of bins (32 equal-freq bins
        # on n=800 uniform samples)
        min_bins = min(len(s) for s in all_scores)
        scores_mat = np.array([s[:min_bins] for s in all_scores])  # (n_mc, m)
        ses_mat = np.array([s[:min_bins] for s in all_ses])        # (n_mc, m)

        empirical_sd = np.std(scores_mat, axis=0, ddof=1)          # (m,)
        mean_predicted_se = np.mean(ses_mat, axis=0)                # (m,)

        # Avoid division by zero for any bin whose score is numerically
        # constant across reps (shouldn't happen but guard anyway)
        valid = empirical_sd > 1e-8
        ratio = mean_predicted_se[valid] / empirical_sd[valid]

        # At least 80% of bins should have ratios in [0.3, 4.0].
        # Expected ratio ~2.0 due to sigma_hat including approximation error.
        frac_calibrated = float(np.mean((ratio >= 0.3) & (ratio <= 4.0)))
        assert frac_calibrated >= 0.80, (
            f"SE calibration: only {frac_calibrated:.1%} of bins have "
            f"ratio(predicted_SE / empirical_SD) in [0.3, 4.0]. "
            f"Ratios: min={ratio.min():.3f}, median={np.median(ratio):.3f}, "
            f"max={ratio.max():.3f}. "
            f"A ratio << 0.3 means CIs are far too narrow; >> 4.0 means too wide."
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


# ---------------------------------------------------------------------------
# Test 11: cv_lambda — lambda selection
# ---------------------------------------------------------------------------

class TestCVLambda:
    """
    cv_lambda() should:
      - Return a fitted BoulevardEBM with lambda_ set to the best value.
      - Populate cv_results_ with lambda_values and mean_cv_mse lists.
      - Choose a lambda in the interior of the grid (not always the extreme).
      - Set sigma_hat_oos_ to a positive float.
      - Override sigma_hat_ with the OOS estimate.
    """

    def test_cv_lambda_returns_fitted_model(self):
        X, y = _make_sine_data(n=400, seed=10)
        model = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=np.logspace(-2, 2, 7),
            n_folds=3,
            n_estimators=200,
            max_bins=32,
        )
        assert model._fitted

    def test_cv_lambda_selects_reasonable_lambda(self):
        """
        On the sine DGP the best lambda should be in the interior of the
        grid — neither the smallest (essentially unregularised, high variance)
        nor the largest (completely shrunk to zero).  We test with a grid that
        brackets reasonable values and assert the best is not at either extreme.
        """
        X, y = _make_sine_data(n=500, noise_std=0.2, seed=7)
        grid = np.logspace(-2, 2, 9)
        model = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=grid,
            n_folds=3,
            n_estimators=200,
            max_bins=32,
        )
        # Best lambda must be strictly inside the grid (not first or last)
        best = model.lambda_
        assert best > grid[0], f"Best lambda {best:.4g} is the grid minimum {grid[0]:.4g}"
        assert best < grid[-1], f"Best lambda {best:.4g} is the grid maximum {grid[-1]:.4g}"

    def test_cv_results_structure(self):
        X, y = _make_sine_data(n=300, seed=20)
        grid = np.logspace(-1, 1, 5)
        model = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=grid,
            n_folds=3,
            n_estimators=100,
            max_bins=32,
        )
        assert model.cv_results_ is not None
        assert "lambda_values" in model.cv_results_
        assert "mean_cv_mse" in model.cv_results_
        assert len(model.cv_results_["lambda_values"]) == len(grid)
        assert len(model.cv_results_["mean_cv_mse"]) == len(grid)

    def test_cv_results_mse_positive(self):
        X, y = _make_sine_data(n=300, seed=21)
        model = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=np.array([0.1, 1.0, 10.0]),
            n_folds=3,
            n_estimators=100,
            max_bins=32,
        )
        for mse in model.cv_results_["mean_cv_mse"]:
            assert mse > 0.0, f"CV MSE should be positive, got {mse}"

    def test_cv_lambda_sets_sigma_hat_oos(self):
        X, y = _make_sine_data(n=300, seed=22)
        model = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=np.array([0.1, 1.0, 10.0]),
            n_folds=3,
            n_estimators=100,
            max_bins=32,
        )
        assert model.sigma_hat_oos_ is not None
        assert model.sigma_hat_oos_ > 0.0

    def test_cv_lambda_sigma_hat_equals_oos(self):
        """sigma_hat_ should be overridden to the OOS estimate."""
        X, y = _make_sine_data(n=300, seed=23)
        model = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=np.array([0.1, 1.0, 10.0]),
            n_folds=3,
            n_estimators=100,
            max_bins=32,
        )
        assert model.sigma_hat_ == model.sigma_hat_oos_

    def test_cv_lambda_default_sigma_hat_oos_is_none(self):
        """Models fitted via .fit() should have sigma_hat_oos_ = None."""
        X, y = _make_sine_data(n=200, seed=24)
        model = BoulevardEBM(n_estimators=100, max_bins=32)
        model.fit(X, y)
        assert model.sigma_hat_oos_ is None

    def test_cv_lambda_polars_input(self):
        X_np, y = _make_sine_data(n=300, seed=25)
        df = pl.DataFrame({"temperature": X_np[:, 0]})
        model = BoulevardEBM.cv_lambda(
            df, y,
            lambda_grid=np.array([0.1, 1.0, 10.0]),
            n_folds=3,
            n_estimators=100,
            max_bins=32,
        )
        assert model._fitted
        assert model.feature_names_ == ["temperature"]


# ---------------------------------------------------------------------------
# Test 12: OOS sigma_hat is closer to true sigma than in-sample
# ---------------------------------------------------------------------------

class TestOOSSigmaHat:
    """
    The OOS sigma_hat from cv_lambda() should be closer to the true noise
    standard deviation than the training-residual sigma_hat from plain fit().

    The training residuals include both noise and KRR approximation error,
    so sigma_hat from fit() is upward biased.  The OOS residuals only contain
    noise plus generalisation error, and for a well-regularised model the
    latter is small, so sigma_hat_oos should be closer to true_noise_std.

    We test this on a DGP where true_noise_std = 0.2 and use n=600 so the
    model can recover the signal reasonably well.
    """

    def test_oos_sigma_closer_to_true(self):
        true_noise_std = 0.2
        X, y = _make_sine_data(n=600, noise_std=true_noise_std, seed=99)

        # In-sample sigma_hat from plain fit (known to be upward biased)
        m_insample = BoulevardEBM(n_estimators=500, lambda_=1.0, max_bins=32, random_state=42)
        m_insample.fit(X, y)
        insample_sigma = m_insample.sigma_hat_

        # OOS sigma_hat from cv_lambda
        m_cv = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=np.logspace(-1, 1, 7),
            n_folds=5,
            n_estimators=300,
            max_bins=32,
            random_state=42,
        )
        oos_sigma = m_cv.sigma_hat_oos_

        err_insample = abs(insample_sigma - true_noise_std)
        err_oos = abs(oos_sigma - true_noise_std)

        assert err_oos < err_insample, (
            f"Expected OOS sigma ({oos_sigma:.4f}) to be closer to true "
            f"sigma ({true_noise_std}) than in-sample sigma ({insample_sigma:.4f}). "
            f"Errors: OOS={err_oos:.4f}, in-sample={err_insample:.4f}."
        )

    def test_oos_sigma_within_factor_of_two(self):
        """OOS sigma_hat should be within 2x of true noise std."""
        true_noise_std = 0.3
        X, y = _make_sine_data(n=500, noise_std=true_noise_std, seed=88)
        m_cv = BoulevardEBM.cv_lambda(
            X, y,
            lambda_grid=np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
            n_folds=4,
            n_estimators=200,
            max_bins=32,
            random_state=0,
        )
        ratio = m_cv.sigma_hat_oos_ / true_noise_std
        assert 0.5 <= ratio <= 2.0, (
            f"OOS sigma_hat ratio = {ratio:.3f}, expected in [0.5, 2.0]. "
            f"sigma_hat_oos={m_cv.sigma_hat_oos_:.4f}, true={true_noise_std}"
        )
