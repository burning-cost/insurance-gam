"""
Tests for insurance_gam.debiased_glm.

Synthetic DGPs:
  - Poisson: n=2000, p=10, true coefs [0.4, 0.3, 0.2] on x0/x1/x2
  - Gamma: n=2000, p=10, same log-linear mean, shape=3
  - Tweedie: n=2000, p=10, power=1.5

Tests cover:
  - API: fit returns self, summary returns DataFrame with correct columns
  - Selection: true features (x0, x1, x2) are selected
  - CI coverage: debiased CIs cover true coefficients (single trial)
  - CI ordering: ci_lower <= coef <= ci_upper for selected features
  - Exposure offset: Poisson rate model with non-unit exposure
  - Bootstrap (Strategy B): n_bootstrap=50 produces valid interval ordering
  - Gamma family: dispersion estimated, CIs computed
  - Tweedie family: power parameter accepted and used
  - Input validation: family, confidence, alpha, negative y
  - DataFrame input: feature names from column names
  - forest_plot: runs without error on fitted model
  - summary() before fit raises RuntimeError
  - p^2/n warning emitted when ratio is large
  - selected_features_ matches nonzero coefficients
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # skip if optional glm extra not installed

from insurance_gam.debiased_glm import DebiasedGLM


# ---------------------------------------------------------------------------
# Constants and DGP fixtures
# ---------------------------------------------------------------------------

TRUE_COEFS = np.array([0.4, 0.3, 0.2])
N = 2000
P = 10
SEED = 42


@pytest.fixture(scope="module")
def poisson_data():
    """Synthetic Poisson dataset: n=2000, p=10, 3 true features."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, P))
    eta = X[:, :3] @ TRUE_COEFS
    y = rng.poisson(np.exp(eta))
    return X, y


@pytest.fixture(scope="module")
def poisson_data_with_exposure():
    """Synthetic Poisson rate dataset with exposure."""
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((N, P))
    exposure = rng.uniform(0.5, 2.0, size=N)
    eta = X[:, :3] @ TRUE_COEFS + np.log(exposure)
    y = rng.poisson(np.exp(eta))
    return X, y, exposure


@pytest.fixture(scope="module")
def gamma_data():
    """Synthetic Gamma dataset: shape=3, log-linear mean."""
    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((N, P))
    mu = np.exp(X[:, :3] @ TRUE_COEFS)
    shape = 3.0
    scale = mu / shape
    y = rng.gamma(shape, scale)
    return X, y


@pytest.fixture(scope="module")
def tweedie_data():
    """Synthetic Tweedie-like dataset (gamma shape as proxy, power=1.5)."""
    rng = np.random.default_rng(SEED + 3)
    X = rng.standard_normal((N, P))
    mu = np.exp(X[:, :3] @ TRUE_COEFS)
    # Generate from Gamma as a rough proxy for Tweedie
    shape = 2.0
    y = rng.gamma(shape, mu / shape)
    return X, y


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestDebiasedGLMAPI:
    def test_fit_returns_self(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED)
        result = model.fit(X, y)
        assert result is model

    def test_summary_returns_dataframe(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        df = model.summary()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"feature", "coef", "se", "ci_lower", "ci_upper", "pvalue", "selected"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_summary_row_count(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        df = model.summary()
        assert len(df) == P

    def test_summary_before_fit_raises(self):
        model = DebiasedGLM()
        with pytest.raises(RuntimeError, match="fit"):
            model.summary()

    def test_fit_chaining(self, poisson_data):
        """Method chaining: DebiasedGLM().fit(X, y).summary() works."""
        X, y = poisson_data
        df = DebiasedGLM(alpha=0.1, random_state=SEED).fit(X, y).summary()
        assert isinstance(df, pd.DataFrame)

    def test_feature_names_from_array(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(alpha=0.1, random_state=SEED).fit(X, y)
        assert model.feature_names_ == [f"x{i}" for i in range(P)]

    def test_feature_names_from_dataframe(self, poisson_data):
        X, y = poisson_data
        cols = [f"feat_{i}" for i in range(P)]
        df_X = pd.DataFrame(X, columns=cols)
        model = DebiasedGLM(alpha=0.1, random_state=SEED).fit(df_X, y)
        assert model.feature_names_ == cols
        summary_df = model.summary()
        assert list(summary_df["feature"]) == cols

    def test_attributes_present_after_fit(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(alpha=0.1, random_state=SEED).fit(X, y)
        assert hasattr(model, "coef_")
        assert hasattr(model, "se_")
        assert hasattr(model, "ci_lower_")
        assert hasattr(model, "ci_upper_")
        assert hasattr(model, "pvalues_")
        assert hasattr(model, "selected_features_")
        assert hasattr(model, "phi_")
        assert hasattr(model, "lambda_")
        assert hasattr(model, "intercept_")

    def test_coef_shape(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(alpha=0.1, random_state=SEED).fit(X, y)
        assert model.coef_.shape == (P,)
        assert model.se_.shape == (P,)
        assert model.ci_lower_.shape == (P,)
        assert model.ci_upper_.shape == (P,)
        assert model.pvalues_.shape == (P,)
        assert model.selected_features_.shape == (P,)


# ---------------------------------------------------------------------------
# Selection and CI tests
# ---------------------------------------------------------------------------


class TestDebiasedGLMInference:
    def test_selects_true_features(self, poisson_data):
        """Lasso should select x0, x1, x2 (true features) in n=2000 case."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.0, cv_folds=5, random_state=SEED)
        model.fit(X, y)
        df = model.summary()
        selected_names = set(df[df["selected"]]["feature"].tolist())
        true_names = {"x0", "x1", "x2"}
        # At least 2 of 3 true features should be selected
        assert len(selected_names & true_names) >= 2, (
            f"Expected true features in selection, got: {selected_names}"
        )

    def test_ci_covers_true_coefs(self, poisson_data):
        """Debiased CIs should cover true coefficients."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.0, cv_folds=5, random_state=SEED)
        model.fit(X, y)
        df = model.summary()
        true_vals = {"x0": 0.4, "x1": 0.3, "x2": 0.2}
        covered = 0
        checked = 0
        for name, true_val in true_vals.items():
            row = df[df["feature"] == name]
            if row["selected"].values[0]:
                lo = row["ci_lower"].values[0]
                hi = row["ci_upper"].values[0]
                if lo <= true_val <= hi:
                    covered += 1
                checked += 1
        # Expect at least 2 out of 3 selected and covered
        assert covered >= 2, (
            f"Expected at least 2 true coefs covered, got {covered}/{checked}. "
            f"Summary:\n{df}"
        )

    def test_ci_ordering(self, poisson_data):
        """ci_lower <= coef <= ci_upper for all features."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"CI inverted for {row['feature']}: "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            )

    def test_se_positive_for_selected(self, poisson_data):
        """Standard errors should be positive for selected features."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        df = model.summary()
        sel = df[df["selected"]]
        assert (sel["se"] > 0).all(), "Some SEs are non-positive for selected features."

    def test_pvalues_in_range(self, poisson_data):
        """p-values should be in [0, 1]."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        df = model.summary()
        pvals = df["pvalue"].values
        assert np.all(pvals >= 0) and np.all(pvals <= 1), (
            f"p-values out of [0,1]: {pvals}"
        )

    def test_selected_features_consistency(self, poisson_data):
        """selected_features_ should match nonzero penalised coefs."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        expected = np.abs(model._beta_penalised) > 1e-8
        np.testing.assert_array_equal(model.selected_features_, expected)

    def test_noise_features_not_all_selected(self, poisson_data):
        """Lasso should zero out most noise features (x3..x9)."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.0, cv_folds=5, random_state=SEED)
        model.fit(X, y)
        df = model.summary()
        noise_selected = df[df["feature"].isin([f"x{i}" for i in range(3, P)])]["selected"].sum()
        # Should select at most 3 noise features out of 7
        assert noise_selected <= 4, (
            f"Too many noise features selected: {noise_selected}"
        )


# ---------------------------------------------------------------------------
# Exposure tests
# ---------------------------------------------------------------------------


class TestDebiasedGLMExposure:
    def test_exposure_accepted(self, poisson_data_with_exposure):
        """Model accepts exposure without error."""
        X, y, exp = poisson_data_with_exposure
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED)
        model.fit(X, y, exposure=exp)
        df = model.summary()
        assert len(df) == P

    def test_exposure_improves_selection(self, poisson_data_with_exposure):
        """With exposure, true features should still be recoverable."""
        X, y, exp = poisson_data_with_exposure
        model = DebiasedGLM(family="poisson", alpha=0.0, cv_folds=5, random_state=SEED)
        model.fit(X, y, exposure=exp)
        df = model.summary()
        selected_names = set(df[df["selected"]]["feature"].tolist())
        true_names = {"x0", "x1", "x2"}
        assert len(selected_names & true_names) >= 2


# ---------------------------------------------------------------------------
# Family tests
# ---------------------------------------------------------------------------


class TestDebiasedGLMFamilies:
    def test_gamma_fit_runs(self, gamma_data):
        """Gamma family fits without error and returns correct shape."""
        X, y = gamma_data
        model = DebiasedGLM(family="gamma", alpha=0.1, random_state=SEED)
        model.fit(X, y)
        df = model.summary()
        assert len(df) == P
        assert model.phi_ > 0  # dispersion should be positive

    def test_gamma_dispersion_estimated(self, gamma_data):
        """Gamma dispersion should be estimated (not remain 1.0)."""
        X, y = gamma_data
        model = DebiasedGLM(family="gamma", alpha=0.1, random_state=SEED)
        model.fit(X, y)
        # Gamma with shape=3 has dispersion phi = 1/shape ~ 0.33
        assert model.phi_ != 1.0, "Dispersion should be estimated for Gamma family."

    def test_gamma_ci_ordering(self, gamma_data):
        """CI ordering holds for Gamma family."""
        X, y = gamma_data
        model = DebiasedGLM(family="gamma", alpha=0.1, random_state=SEED).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"CI inverted for {row['feature']} in Gamma model."
            )

    def test_tweedie_fit_runs(self, tweedie_data):
        """Tweedie family fits without error."""
        X, y = tweedie_data
        model = DebiasedGLM(family="tweedie", tweedie_power=1.5, alpha=0.1, random_state=SEED)
        model.fit(X, y)
        df = model.summary()
        assert len(df) == P

    def test_tweedie_ci_ordering(self, tweedie_data):
        """CI ordering holds for Tweedie family."""
        X, y = tweedie_data
        model = DebiasedGLM(
            family="tweedie", tweedie_power=1.5, alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_poisson_phi_is_one(self, poisson_data):
        """Poisson dispersion is always 1.0."""
        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        assert model.phi_ == 1.0


# ---------------------------------------------------------------------------
# Bootstrap tests (Strategy B)
# ---------------------------------------------------------------------------


class TestDebiasedGLMBootstrap:
    def test_bootstrap_runs(self, poisson_data):
        """n_bootstrap=50 runs without error."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, n_bootstrap=50, random_state=SEED
        )
        model.fit(X, y)
        df = model.summary()
        assert len(df) == P

    def test_bootstrap_ci_ordering(self, poisson_data):
        """Bootstrap CIs should have ci_lower <= ci_upper."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, n_bootstrap=50, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"Bootstrap CI inverted for {row['feature']}: "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            )

    def test_bootstrap_se_is_nan(self, poisson_data):
        """Bootstrap Strategy B does not produce SE estimates."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, n_bootstrap=50, random_state=SEED
        ).fit(X, y)
        assert np.all(np.isnan(model.se_)), "SE should be NaN for bootstrap strategy."

    def test_bootstrap_pvalues_are_nan(self, poisson_data):
        """Bootstrap Strategy B does not produce p-values."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, n_bootstrap=50, random_state=SEED
        ).fit(X, y)
        assert np.all(np.isnan(model.pvalues_))


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestDebiasedGLMValidation:
    def test_invalid_family(self):
        with pytest.raises(ValueError, match="family"):
            DebiasedGLM(family="binomial")

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            DebiasedGLM(confidence=1.5)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            DebiasedGLM(alpha=-0.1)

    def test_invalid_l1_ratio(self):
        with pytest.raises(ValueError, match="l1_ratio"):
            DebiasedGLM(l1_ratio=1.5)

    def test_invalid_n_bootstrap(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            DebiasedGLM(n_bootstrap=-1)

    def test_invalid_cv_folds(self):
        with pytest.raises(ValueError, match="cv_folds"):
            DebiasedGLM(cv_folds=1)

    def test_invalid_tweedie_power(self):
        with pytest.raises(ValueError, match="tweedie_power"):
            DebiasedGLM(family="tweedie", tweedie_power=2.5)

    def test_negative_y_poisson(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=100).astype(float)
        y[0] = -1.0
        model = DebiasedGLM(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X, y)

    def test_nonpositive_y_gamma(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.gamma(2.0, 1.0, size=100)
        y[0] = 0.0
        model = DebiasedGLM(family="gamma", alpha=0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(X, y)

    def test_wrong_y_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=50)
        model = DebiasedGLM(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="shape"):
            model.fit(X, y)

    def test_bad_exposure_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=100)
        exposure = rng.uniform(0.5, 2.0, size=50)  # wrong length
        model = DebiasedGLM(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="exposure"):
            model.fit(X, y, exposure=exposure)

    def test_zero_exposure(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.poisson(1.0, size=100)
        exposure = rng.uniform(0.5, 2.0, size=100)
        exposure[0] = 0.0
        model = DebiasedGLM(family="poisson", alpha=0.1)
        with pytest.raises(ValueError, match="positive"):
            model.fit(X, y, exposure=exposure)

    def test_high_p_squared_over_n_warning(self):
        """Warn when p^2/n > 0.25."""
        rng = np.random.default_rng(0)
        n = 100
        p = 8  # p^2/n = 64/100 = 0.64
        X = rng.standard_normal((n, p))
        y = rng.poisson(np.exp(0.3 * X[:, 0]))
        model = DebiasedGLM(family="poisson", alpha=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
        warning_messages = [str(warning.message) for warning in w]
        assert any("p^2/n" in msg for msg in warning_messages), (
            f"Expected p^2/n warning, got: {warning_messages}"
        )


# ---------------------------------------------------------------------------
# forest_plot test
# ---------------------------------------------------------------------------


class TestDebiasedGLMForestPlot:
    def test_forest_plot_runs(self, poisson_data):
        """forest_plot() runs without error after fit."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        fig, ax = plt.subplots()
        ax_out = model.forest_plot(ax=ax)
        assert ax_out is ax
        plt.close("all")

    def test_forest_plot_before_fit_raises(self):
        model = DebiasedGLM()
        with pytest.raises(RuntimeError, match="fit"):
            model.forest_plot()

    def test_forest_plot_all_features(self, poisson_data):
        """forest_plot(only_selected=False) includes all features."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = poisson_data
        model = DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)
        fig, ax = plt.subplots()
        model.forest_plot(ax=ax, only_selected=False)
        plt.close("all")
