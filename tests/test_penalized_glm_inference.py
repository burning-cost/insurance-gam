"""
Tests for insurance_gam.penalized_glm_inference.PenalizedGLMInference.

Synthetic DGPs:
  - Poisson: n=2000, p=10, true coefs [0.4, 0.3, 0.2] on x0/x1/x2
  - Gamma: n=2000, p=10, same log-linear mean, shape=3
  - Tweedie: n=2000, p=10, power=1.5 (Gamma proxy)

Tests cover:
  - fit() returns self (method chaining)
  - confidence_intervals() returns DataFrame with correct columns
  - bootstrap_ci() returns DataFrame with correct columns
  - summary() is an alias for confidence_intervals(alpha=0.05)
  - CI ordering: ci_lower <= ci_upper for all features, both strategies
  - CI covers true coefficients for selected features (single-trial)
  - SE is positive for selected features (asymptotic)
  - SE is NaN for bootstrap strategy
  - pvalue is NaN for bootstrap strategy
  - p-values in [0, 1] for asymptotic strategy
  - selected_features_ consistency with non-zero coef_penalized_
  - coef_penalized_ stored on the fitted object
  - alpha parameter changes CI width (smaller alpha = wider CIs)
  - exposure offset accepted and produces valid output
  - DataFrame input: column names used as feature names
  - feature names from array input: "x0", "x1", ...
  - Gamma family: dispersion estimated and positive
  - Tweedie family: tweedie_power accepted
  - Poisson dispersion is 1.0
  - p^2/n warning emitted when ratio > 0.25
  - forest_plot() runs with asymptotic strategy
  - forest_plot() runs with bootstrap strategy
  - forest_plot() before fit raises RuntimeError
  - forest_plot() with invalid strategy raises ValueError
  - Input validation: invalid family, alpha, l1_ratio, cv_folds, tweedie_power
  - Negative y for Poisson raises ValueError
  - Non-positive y for Gamma raises ValueError
  - Wrong y shape raises ValueError
  - Bad exposure shape raises ValueError
  - Zero exposure raises ValueError
  - confidence_intervals() before fit raises RuntimeError
  - bootstrap_ci() before fit raises RuntimeError
  - bootstrap_ci() with n_bootstrap < 1 raises ValueError
  - confidence_intervals() with alpha out of range raises ValueError
  - lambda_ attribute positive after fit
  - n_obs_ and n_features_ match input dimensions
  - bootstrap: coef column equals coef_penalized_ (not debiased)
  - asymptotic: coef column differs from coef_penalized_ for selected features
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_gam.penalized_glm_inference import PenalizedGLMInference


# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

TRUE_COEFS = np.array([0.4, 0.3, 0.2])
N = 2000
P = 10
SEED = 42
EXPECTED_COLS = {
    "feature",
    "coef_penalized",
    "coef",
    "se",
    "ci_lower",
    "ci_upper",
    "pvalue",
    "selected",
}


@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, P))
    eta = X[:, :3] @ TRUE_COEFS
    y = rng.poisson(np.exp(eta))
    return X, y


@pytest.fixture(scope="module")
def poisson_data_with_exposure():
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((N, P))
    exposure = rng.uniform(0.5, 2.0, size=N)
    eta = X[:, :3] @ TRUE_COEFS + np.log(exposure)
    y = rng.poisson(np.exp(eta))
    return X, y, exposure


@pytest.fixture(scope="module")
def gamma_data():
    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((N, P))
    mu = np.exp(X[:, :3] @ TRUE_COEFS)
    shape = 3.0
    scale = mu / shape
    y = rng.gamma(shape, scale)
    return X, y


@pytest.fixture(scope="module")
def tweedie_data():
    rng = np.random.default_rng(SEED + 3)
    X = rng.standard_normal((N, P))
    mu = np.exp(X[:, :3] @ TRUE_COEFS)
    shape = 2.0
    y = rng.gamma(shape, mu / shape)
    return X, y


@pytest.fixture(scope="module")
def fitted_poisson(poisson_data):
    """Pre-fitted Poisson model (module scope for speed)."""
    X, y = poisson_data
    m = PenalizedGLMInference(
        family="poisson", alpha=0.1, random_state=SEED
    ).fit(X, y)
    return m


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestFitAPI:
    def test_fit_returns_self(self, poisson_data):
        X, y = poisson_data
        m = PenalizedGLMInference(family="poisson", alpha=0.1, random_state=SEED)
        result = m.fit(X, y)
        assert result is m

    def test_method_chaining(self, poisson_data):
        X, y = poisson_data
        df = (
            PenalizedGLMInference(alpha=0.1, random_state=SEED)
            .fit(X, y)
            .confidence_intervals()
        )
        assert isinstance(df, pd.DataFrame)

    def test_attributes_set_after_fit(self, fitted_poisson):
        m = fitted_poisson
        assert hasattr(m, "coef_penalized_")
        assert hasattr(m, "intercept_")
        assert hasattr(m, "selected_features_")
        assert hasattr(m, "phi_")
        assert hasattr(m, "lambda_")
        assert hasattr(m, "n_obs_")
        assert hasattr(m, "n_features_")
        assert hasattr(m, "feature_names_")

    def test_n_obs_correct(self, poisson_data, fitted_poisson):
        X, _ = poisson_data
        assert fitted_poisson.n_obs_ == X.shape[0]

    def test_n_features_correct(self, poisson_data, fitted_poisson):
        X, _ = poisson_data
        assert fitted_poisson.n_features_ == X.shape[1]

    def test_lambda_positive(self, fitted_poisson):
        assert fitted_poisson.lambda_ > 0

    def test_coef_penalized_shape(self, fitted_poisson):
        assert fitted_poisson.coef_penalized_.shape == (P,)

    def test_selected_features_shape(self, fitted_poisson):
        assert fitted_poisson.selected_features_.shape == (P,)

    def test_feature_names_from_array(self, fitted_poisson):
        assert fitted_poisson.feature_names_ == [f"x{i}" for i in range(P)]

    def test_feature_names_from_dataframe(self, poisson_data):
        X, y = poisson_data
        cols = [f"feat_{i}" for i in range(P)]
        df_X = pd.DataFrame(X, columns=cols)
        m = PenalizedGLMInference(alpha=0.1, random_state=SEED).fit(df_X, y)
        assert m.feature_names_ == cols
        df = m.confidence_intervals()
        assert list(df["feature"]) == cols

    def test_selected_features_matches_nonzero(self, fitted_poisson):
        expected = np.abs(fitted_poisson.coef_penalized_) > 1e-8
        np.testing.assert_array_equal(fitted_poisson.selected_features_, expected)


# ---------------------------------------------------------------------------
# confidence_intervals() tests (Strategy A)
# ---------------------------------------------------------------------------


class TestAsymptoticCI:
    def test_returns_dataframe(self, fitted_poisson):
        df = fitted_poisson.confidence_intervals()
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_columns(self, fitted_poisson):
        df = fitted_poisson.confidence_intervals()
        assert EXPECTED_COLS.issubset(set(df.columns))

    def test_row_count(self, fitted_poisson):
        df = fitted_poisson.confidence_intervals()
        assert len(df) == P

    def test_ci_ordering(self, fitted_poisson):
        df = fitted_poisson.confidence_intervals()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"CI inverted for {row['feature']}: "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            )

    def test_se_positive_for_selected(self, fitted_poisson):
        df = fitted_poisson.confidence_intervals()
        sel = df[df["selected"]]
        assert (sel["se"] > 0).all()

    def test_pvalues_in_range(self, fitted_poisson):
        df = fitted_poisson.confidence_intervals()
        pvals = df["pvalue"].values
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_coef_differs_from_penalized_for_selected(self, poisson_data):
        """Debiasing should change the coefficient relative to the penalized estimate."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.0, cv_folds=5, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        sel = df[df["selected"]]
        # At least one selected feature should have a different debiased coef
        diffs = np.abs(sel["coef"].values - sel["coef_penalized"].values)
        assert np.any(diffs > 1e-10), (
            "Debiasing had no effect on any selected coefficient."
        )

    def test_selects_true_features(self, poisson_data):
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.0, cv_folds=5, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        selected = set(df[df["selected"]]["feature"].tolist())
        true_names = {"x0", "x1", "x2"}
        assert len(selected & true_names) >= 2, (
            f"Expected at least 2 true features selected, got: {selected}"
        )

    def test_ci_covers_true_coefs(self, poisson_data):
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.0, cv_folds=5, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals(alpha=0.05)
        true_vals = {"x0": 0.4, "x1": 0.3, "x2": 0.2}
        covered = sum(
            1
            for name, val in true_vals.items()
            if (row := df[df["feature"] == name]).iloc[0]["selected"]
            and row.iloc[0]["ci_lower"] <= val <= row.iloc[0]["ci_upper"]
        )
        assert covered >= 2, f"Expected >= 2 true coefs covered, got {covered}"

    def test_narrower_alpha_gives_wider_ci(self, fitted_poisson):
        """Smaller alpha (higher confidence) should give wider CIs."""
        df_95 = fitted_poisson.confidence_intervals(alpha=0.05)
        df_99 = fitted_poisson.confidence_intervals(alpha=0.01)
        # Take width for selected features
        sel_95 = df_95[df_95["selected"]]
        sel_99 = df_99[df_99["selected"]]
        if not sel_95.empty:
            widths_95 = (sel_95["ci_upper"] - sel_95["ci_lower"]).values
            widths_99 = (sel_99["ci_upper"] - sel_99["ci_lower"]).values
            assert np.all(widths_99 >= widths_95 - 1e-10), (
                "99% CIs should be at least as wide as 95% CIs."
            )

    def test_before_fit_raises(self):
        m = PenalizedGLMInference()
        with pytest.raises(RuntimeError, match="fit"):
            m.confidence_intervals()

    def test_invalid_alpha_raises(self, fitted_poisson):
        with pytest.raises(ValueError, match="alpha"):
            fitted_poisson.confidence_intervals(alpha=1.5)

    def test_zero_alpha_raises(self, fitted_poisson):
        with pytest.raises(ValueError, match="alpha"):
            fitted_poisson.confidence_intervals(alpha=0.0)


# ---------------------------------------------------------------------------
# bootstrap_ci() tests (Strategy B)
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_dataframe(self, fitted_poisson):
        df = fitted_poisson.bootstrap_ci(alpha=0.05, n_bootstrap=50)
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_columns(self, fitted_poisson):
        df = fitted_poisson.bootstrap_ci(n_bootstrap=50)
        assert EXPECTED_COLS.issubset(set(df.columns))

    def test_row_count(self, fitted_poisson):
        df = fitted_poisson.bootstrap_ci(n_bootstrap=50)
        assert len(df) == P

    def test_ci_ordering(self, fitted_poisson):
        df = fitted_poisson.bootstrap_ci(n_bootstrap=50, random_state=SEED)
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"Bootstrap CI inverted for {row['feature']}: "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            )

    def test_se_is_nan(self, fitted_poisson):
        df = fitted_poisson.bootstrap_ci(n_bootstrap=50)
        assert np.all(np.isnan(df["se"].values))

    def test_pvalue_is_nan(self, fitted_poisson):
        df = fitted_poisson.bootstrap_ci(n_bootstrap=50)
        assert np.all(np.isnan(df["pvalue"].values))

    def test_coef_equals_coef_penalized(self, fitted_poisson):
        """Bootstrap point estimate is the penalized estimate (bias in CI bounds)."""
        df = fitted_poisson.bootstrap_ci(n_bootstrap=50)
        np.testing.assert_allclose(
            df["coef"].values,
            df["coef_penalized"].values,
            rtol=0,
            atol=1e-12,
        )

    def test_before_fit_raises(self):
        m = PenalizedGLMInference()
        with pytest.raises(RuntimeError, match="fit"):
            m.bootstrap_ci()

    def test_n_bootstrap_zero_raises(self, fitted_poisson):
        with pytest.raises(ValueError, match="n_bootstrap"):
            fitted_poisson.bootstrap_ci(n_bootstrap=0)

    def test_invalid_alpha_raises(self, fitted_poisson):
        with pytest.raises(ValueError, match="alpha"):
            fitted_poisson.bootstrap_ci(alpha=0.0)

    def test_random_state_override(self, fitted_poisson):
        """random_state argument overrides instance seed for reproducibility."""
        df1 = fitted_poisson.bootstrap_ci(n_bootstrap=30, random_state=0)
        df2 = fitted_poisson.bootstrap_ci(n_bootstrap=30, random_state=0)
        np.testing.assert_allclose(
            df1["ci_lower"].values, df2["ci_lower"].values, rtol=1e-8
        )


# ---------------------------------------------------------------------------
# summary() tests
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_equivalent_to_ci_alpha05(self, fitted_poisson):
        df_sum = fitted_poisson.summary()
        df_ci = fitted_poisson.confidence_intervals(alpha=0.05)
        pd.testing.assert_frame_equal(df_sum, df_ci)

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            PenalizedGLMInference().summary()

    def test_summary_custom_alpha(self, fitted_poisson):
        df = fitted_poisson.summary(alpha=0.01)
        df_ci = fitted_poisson.confidence_intervals(alpha=0.01)
        pd.testing.assert_frame_equal(df, df_ci)


# ---------------------------------------------------------------------------
# Exposure tests
# ---------------------------------------------------------------------------


class TestExposure:
    def test_exposure_accepted(self, poisson_data_with_exposure):
        X, y, exp = poisson_data_with_exposure
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y, exposure=exp)
        df = m.confidence_intervals()
        assert len(df) == P

    def test_exposure_ci_ordering(self, poisson_data_with_exposure):
        X, y, exp = poisson_data_with_exposure
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y, exposure=exp)
        df = m.confidence_intervals()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]


# ---------------------------------------------------------------------------
# Family tests
# ---------------------------------------------------------------------------


class TestFamilies:
    def test_poisson_phi_is_one(self, fitted_poisson):
        assert fitted_poisson.phi_ == 1.0

    def test_gamma_dispersion_estimated(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert m.phi_ > 0
        assert m.phi_ != 1.0, "Gamma dispersion should be estimated."

    def test_gamma_ci_ordering(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_gamma_row_count(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert len(m.confidence_intervals()) == P

    def test_tweedie_fit_runs(self, tweedie_data):
        X, y = tweedie_data
        m = PenalizedGLMInference(
            family="tweedie", tweedie_power=1.5, alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        assert len(df) == P

    def test_tweedie_ci_ordering(self, tweedie_data):
        X, y = tweedie_data
        m = PenalizedGLMInference(
            family="tweedie", tweedie_power=1.5, alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]


# ---------------------------------------------------------------------------
# Noise feature selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_noise_features_mostly_excluded(self, poisson_data):
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.0, cv_folds=5, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        noise = df[df["feature"].isin([f"x{i}" for i in range(3, P)])]
        n_noise_selected = noise["selected"].sum()
        assert n_noise_selected <= 4, (
            f"Too many noise features selected: {n_noise_selected}/7"
        )


# ---------------------------------------------------------------------------
# p^2/n warning
# ---------------------------------------------------------------------------


class TestDimensionWarning:
    def test_high_p_squared_over_n_warns(self):
        rng = np.random.default_rng(0)
        n, p = 100, 8  # p^2/n = 64/100 = 0.64 > 0.25
        X = rng.standard_normal((n, p))
        y = rng.poisson(np.exp(0.3 * X[:, 0]))
        m = PenalizedGLMInference(family="poisson", alpha=0.5).fit(X, y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.confidence_intervals()
        messages = [str(x.message) for x in w]
        assert any("p^2/n" in msg for msg in messages), (
            f"Expected p^2/n warning. Got: {messages}"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_family(self):
        with pytest.raises(ValueError, match="family"):
            PenalizedGLMInference(family="binomial")

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            PenalizedGLMInference(alpha=-0.1)

    def test_invalid_l1_ratio(self):
        with pytest.raises(ValueError, match="l1_ratio"):
            PenalizedGLMInference(l1_ratio=1.5)

    def test_invalid_cv_folds(self):
        with pytest.raises(ValueError, match="cv_folds"):
            PenalizedGLMInference(cv_folds=1)

    def test_invalid_tweedie_power_too_high(self):
        with pytest.raises(ValueError, match="tweedie_power"):
            PenalizedGLMInference(family="tweedie", tweedie_power=2.5)

    def test_invalid_tweedie_power_too_low(self):
        with pytest.raises(ValueError, match="tweedie_power"):
            PenalizedGLMInference(family="tweedie", tweedie_power=0.5)

    def test_negative_y_poisson(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=100).astype(float)
        y[5] = -1.0
        m = PenalizedGLMInference(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="non-negative"):
            m.fit(X, y)

    def test_nonpositive_y_gamma(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.gamma(2.0, 1.0, size=100)
        y[0] = 0.0
        m = PenalizedGLMInference(family="gamma", alpha=0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(X, y)

    def test_wrong_y_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=50)
        m = PenalizedGLMInference(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="shape"):
            m.fit(X, y)

    def test_bad_exposure_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=100)
        exposure = rng.uniform(0.5, 2.0, size=50)
        m = PenalizedGLMInference(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="exposure"):
            m.fit(X, y, exposure=exposure)

    def test_zero_exposure(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=100)
        exp = rng.uniform(0.5, 2.0, size=100)
        exp[0] = 0.0
        m = PenalizedGLMInference(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="positive"):
            m.fit(X, y, exposure=exp)


# ---------------------------------------------------------------------------
# forest_plot tests
# ---------------------------------------------------------------------------


class TestForestPlot:
    def test_asymptotic_plot_runs(self, fitted_poisson):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax_out = fitted_poisson.forest_plot(strategy="asymptotic", ax=ax)
        assert ax_out is ax
        plt.close("all")

    def test_bootstrap_plot_runs(self, fitted_poisson):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax_out = fitted_poisson.forest_plot(
            strategy="bootstrap", n_bootstrap=30, ax=ax
        )
        assert ax_out is ax
        plt.close("all")

    def test_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            PenalizedGLMInference().forest_plot()

    def test_invalid_strategy_raises(self, fitted_poisson):
        with pytest.raises(ValueError, match="strategy"):
            fitted_poisson.forest_plot(strategy="invalid")

    def test_plot_all_features(self, fitted_poisson):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fitted_poisson.forest_plot(only_selected=False, ax=ax)
        plt.close("all")
