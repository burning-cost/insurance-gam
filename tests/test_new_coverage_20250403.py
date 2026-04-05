"""
New coverage tests for insurance-gam (2025-04-03).

Focus areas:
  - PostSelectionGLM: cv_folds validation, subsample param, forest_plot no
    selected raises, DataSplitPostSelectionGLM max_features, forest_plot before
    fit, invalid cv_folds, _bisect helper, _truncated_normal_ci no truncation
  - DebiasedGLM: ridge=0.0 (pure Lasso), forest_plot only_selected=False,
    gamma bootstrap, tweedie bootstrap, forest_plot before fit, l1_ratio
    boundary (1.0 and 0.0)
  - PenalizedGLMInference: gamma bootstrap_ci, tweedie bootstrap_ci,
    two-feature model, bootstrap_ci gamma CI ordering, forest_plot all features,
    summary after bootstrap_ci still works, confidence_intervals multiple alpha
    values produce nested CIs, l1_ratio=1.0 (pure Lasso)
  - Internal helpers: _make_offset, _validate_family, _poisson_pseudo_data,
    _check_family (debiased_glm), _hessian_weights, _sqrt_variance
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # skip if optional glm extra not installed

from insurance_gam.post_selection import (
    PostSelectionGLM,
    DataSplitPostSelectionGLM,
    _bisect,
    _truncated_normal_ci,
    _make_offset,
    _poisson_pseudo_data,
    _validate_family,
)
from insurance_gam.debiased_glm import (
    DebiasedGLM,
    _check_family,
    _hessian_weights,
    _sqrt_variance,
    _estimate_dispersion,
    _variance_function,
)
from insurance_gam.penalized_glm_inference import PenalizedGLMInference


# ---------------------------------------------------------------------------
# Shared DGP constants
# ---------------------------------------------------------------------------

TRUE_COEFS = np.array([0.4, 0.3, 0.2])
N = 2000
P = 10
SEED = 42


@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, P))
    eta = X[:, :3] @ TRUE_COEFS
    y = rng.poisson(np.exp(eta))
    return X, y


@pytest.fixture(scope="module")
def gamma_data():
    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((N, P))
    mu = np.exp(X[:, :3] @ TRUE_COEFS)
    shape = 3.0
    y = rng.gamma(shape, mu / shape)
    return X, y


@pytest.fixture(scope="module")
def tweedie_data():
    rng = np.random.default_rng(SEED + 3)
    X = rng.standard_normal((N, P))
    mu = np.exp(X[:, :3] @ TRUE_COEFS)
    y = rng.gamma(2.0, mu / 2.0)
    return X, y


@pytest.fixture(scope="module")
def fitted_psglm(poisson_data):
    X, y = poisson_data
    return PostSelectionGLM(random_state=SEED).fit(X, y)


@pytest.fixture(scope="module")
def fitted_debiased(poisson_data):
    X, y = poisson_data
    return DebiasedGLM(family="poisson", alpha=0.1, random_state=SEED).fit(X, y)


@pytest.fixture(scope="module")
def fitted_pin(poisson_data):
    X, y = poisson_data
    return PenalizedGLMInference(
        family="poisson", alpha=0.1, random_state=SEED
    ).fit(X, y)


# ---------------------------------------------------------------------------
# Internal helper tests — post_selection
# ---------------------------------------------------------------------------


class TestMakeOffset:
    def test_none_returns_zeros(self):
        offset = _make_offset(None, 5)
        np.testing.assert_array_equal(offset, np.zeros(5))

    def test_log_exposure(self):
        exp = np.array([1.0, np.e, np.e ** 2])
        offset = _make_offset(exp, 3)
        np.testing.assert_allclose(offset, [0.0, 1.0, 2.0], atol=1e-10)

    def test_zero_exposure_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _make_offset(np.array([1.0, 0.0, 1.0]), 3)

    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _make_offset(np.array([1.0, -0.5]), 2)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _make_offset(np.ones(5), 3)


class TestValidateFamily:
    def test_poisson_accepted(self):
        _validate_family("poisson")  # no exception

    def test_gamma_rejected(self):
        with pytest.raises(ValueError, match="poisson"):
            _validate_family("gamma")

    def test_tweedie_rejected(self):
        with pytest.raises(ValueError, match="poisson"):
            _validate_family("tweedie")


class TestPoissonPseudoData:
    def test_pseudo_data_shapes(self):
        rng = np.random.default_rng(0)
        n, p = 50, 4
        X = rng.standard_normal((n, p))
        y = rng.poisson(1.5, size=n).astype(float)
        eta = np.log(y + 0.01)
        z, Z = _poisson_pseudo_data(X, y, eta)
        assert z.shape == (n,)
        assert Z.shape == (n, p)

    def test_pseudo_data_Z_proportional_to_sqrt_mu(self):
        """Z = sqrt(mu) * X — check column norms."""
        rng = np.random.default_rng(1)
        n, p = 30, 3
        X = rng.standard_normal((n, p))
        y = rng.poisson(2.0, size=n).astype(float)
        eta = np.log(y + 0.01)
        _, Z = _poisson_pseudo_data(X, y, eta)
        mu = np.exp(eta)
        sqrt_mu = np.sqrt(np.clip(mu, 1e-10, None))
        expected_Z = sqrt_mu[:, np.newaxis] * X
        np.testing.assert_allclose(Z, expected_Z, atol=1e-10)


class TestBisect:
    def test_finds_root(self):
        x = _bisect(lambda t: t, 0.5, 0.0, 1.0)
        assert abs(x - 0.5) < 1e-5

    def test_quadratic_root(self):
        # f(x) = x^2 - 2, root at sqrt(2) ~ 1.4142
        x = _bisect(lambda t: t ** 2 - 2.0, 0.0, 0.0, 3.0)
        assert abs(x - np.sqrt(2)) < 1e-5

    def test_expands_bracket(self):
        # Target is 0.99, which is outside initial bracket [0, 0.5]
        # f is linear so bracket can be found after expansion
        x = _bisect(lambda t: t, 0.99, 0.0, 0.5)
        assert abs(x - 0.99) < 1e-4


class TestTruncatedNormalCI:
    def test_full_real_line_gives_standard_normal_ci(self):
        """With no truncation, should recover standard normal CI."""
        intervals = [(-np.inf, np.inf)]
        ci_lo, ci_hi, pval = _truncated_normal_ci(intervals, mu=0.5, sigma=0.1, alpha=0.05)
        # Should be ~ 0.5 +/- 1.96 * 0.1
        assert abs(ci_lo - (0.5 - 1.96 * 0.1)) < 0.01
        assert abs(ci_hi - (0.5 + 1.96 * 0.1)) < 0.01

    def test_empty_intervals_gives_standard_normal_ci(self):
        intervals = []
        ci_lo, ci_hi, _ = _truncated_normal_ci(intervals, mu=0.0, sigma=1.0, alpha=0.05)
        assert abs(ci_lo - (-1.96)) < 0.01
        assert abs(ci_hi - 1.96) < 0.01

    def test_ci_ordering(self):
        """ci_lower < ci_upper for reasonable inputs."""
        intervals = [(0.1, 5.0)]
        ci_lo, ci_hi, _ = _truncated_normal_ci(intervals, mu=1.0, sigma=0.5, alpha=0.05)
        assert ci_lo < ci_hi

    def test_pvalue_in_range(self):
        intervals = [(-np.inf, np.inf)]
        _, _, pval = _truncated_normal_ci(intervals, mu=2.0, sigma=1.0, alpha=0.05)
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# PostSelectionGLM — additional edge cases
# ---------------------------------------------------------------------------


class TestPostSelectionGLMExtended:
    def test_invalid_cv_folds_raises(self):
        with pytest.raises(ValueError, match="cv_folds"):
            PostSelectionGLM(cv_folds=1)

    def test_invalid_subsample_raises(self):
        with pytest.raises(ValueError, match="subsample"):
            PostSelectionGLM(subsample=50)

    def test_forest_plot_no_selected_raises(self, poisson_data):
        """forest_plot should raise if no features were selected."""
        import matplotlib
        matplotlib.use("Agg")

        X, y = poisson_data
        # Use an absurdly high alpha to force zero selection
        model = PostSelectionGLM(max_features=0, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        # If Lasso selected nothing, forest_plot should raise
        df = model.summary()
        if not df["selected"].any():
            with pytest.raises(ValueError):
                model.forest_plot()

    def test_summary_has_p_features(self, fitted_psglm):
        df = fitted_psglm.summary()
        assert len(df) == P

    def test_selected_col_is_bool(self, fitted_psglm):
        df = fitted_psglm.summary()
        assert df["selected"].dtype == bool

    def test_non_selected_rows_have_nan_coefficient(self, fitted_psglm):
        df = fitted_psglm.summary()
        non_sel = df[~df["selected"]]
        assert non_sel["coefficient"].isna().all()

    def test_pvalue_in_range_for_selected(self, fitted_psglm):
        df = fitted_psglm.summary()
        sel = df[df["selected"]]
        pvals = sel["pvalue"].dropna().values
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_std_error_positive_for_selected(self, fitted_psglm):
        df = fitted_psglm.summary()
        sel = df[df["selected"]].dropna(subset=["std_error"])
        if len(sel) > 0:
            assert (sel["std_error"] > 0).all()


class TestDataSplitGLMExtended:
    def test_cv_folds_validation(self):
        with pytest.raises(ValueError, match="cv_folds"):
            DataSplitPostSelectionGLM(cv_folds=1)

    def test_alpha_validation(self):
        with pytest.raises(ValueError, match="alpha"):
            DataSplitPostSelectionGLM(alpha=0.0)

    def test_max_features_constraint(self, poisson_data):
        """max_features should limit selection."""
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(max_features=2, random_state=SEED).fit(X, y)
        df = model.summary()
        assert df["selected"].sum() <= 2

    def test_forest_plot_before_fit_raises(self):
        model = DataSplitPostSelectionGLM()
        with pytest.raises(RuntimeError, match="fit"):
            model.forest_plot()

    def test_feature_names_from_dataframe(self, poisson_data):
        X, y = poisson_data
        cols = [f"col_{i}" for i in range(P)]
        df_X = pd.DataFrame(X, columns=cols)
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(df_X, y)
        df = model.summary()
        assert set(df["feature"]) == set(cols)

    def test_pvalue_finite_for_selected(self, poisson_data):
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        sel = df[df["selected"]]
        assert sel["pvalue"].notna().all()

    def test_ci_lower_less_than_ci_upper_for_selected(self, poisson_data):
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        sel = df[df["selected"]]
        assert (sel["ci_lower"] < sel["ci_upper"]).all()


# ---------------------------------------------------------------------------
# DebiasedGLM — additional edge cases
# ---------------------------------------------------------------------------


class TestDebiasedGLMExtended:
    def test_pure_lasso_l1_ratio_one(self, poisson_data):
        """l1_ratio=1.0 should work (pure Lasso)."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, l1_ratio=1.0, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        assert len(df) == P

    def test_ridge_l1_ratio_zero(self, poisson_data):
        """l1_ratio=0.0 should work (pure Ridge — no Lasso sparsity)."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, l1_ratio=0.0, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        assert len(df) == P

    def test_forest_plot_only_selected_false(self, fitted_debiased):
        """forest_plot(only_selected=False) should include all features."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax_out = fitted_debiased.forest_plot(ax=ax, only_selected=False)
        assert ax_out is ax
        plt.close("all")

    def test_forest_plot_no_features_raises(self, poisson_data):
        """forest_plot with only_selected=True but no selected features should raise."""
        import matplotlib
        matplotlib.use("Agg")

        rng = np.random.default_rng(99)
        X = rng.standard_normal((200, 4))
        y = rng.poisson(1.0, size=200)
        # Use very high alpha to force all coefficients to zero
        model = DebiasedGLM(family="poisson", alpha=100.0, random_state=99).fit(X, y)
        df = model.summary()
        if not df["selected"].any():
            with pytest.raises(ValueError):
                model.forest_plot(only_selected=True)

    def test_gamma_bootstrap_ci_ordering(self, gamma_data):
        """Bootstrap CIs for Gamma should maintain ci_lower <= ci_upper."""
        X, y = gamma_data
        model = DebiasedGLM(
            family="gamma", alpha=0.1, n_bootstrap=30, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"Gamma bootstrap CI inverted for {row['feature']}"
            )

    def test_tweedie_bootstrap_ci_ordering(self, tweedie_data):
        """Bootstrap CIs for Tweedie should maintain ci_lower <= ci_upper."""
        X, y = tweedie_data
        model = DebiasedGLM(
            family="tweedie", tweedie_power=1.5, alpha=0.1, n_bootstrap=20,
            random_state=SEED,
        ).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_cv_alpha_zero(self, poisson_data):
        """alpha=0.0 should trigger cross-validation for lambda."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)
        assert model.lambda_ > 0

    def test_confidence_level_90(self, poisson_data):
        """confidence=0.9 should produce narrower CIs than confidence=0.99."""
        X, y = poisson_data
        m90 = DebiasedGLM(
            family="poisson", alpha=0.1, confidence=0.90, random_state=SEED
        ).fit(X, y)
        m99 = DebiasedGLM(
            family="poisson", alpha=0.1, confidence=0.99, random_state=SEED
        ).fit(X, y)
        # For the same selected features, 90% CIs should be narrower than 99%
        w90 = m90.ci_upper_ - m90.ci_lower_
        w99 = m99.ci_upper_ - m99.ci_lower_
        assert np.all(w90 <= w99 + 1e-10)

    def test_intercept_finite(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert np.isfinite(model.intercept_)

    def test_phi_one_for_poisson_always(self, poisson_data):
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert model.phi_ == 1.0

    def test_tweedie_power_boundary_near_one(self, tweedie_data):
        """Tweedie power=1.0 (Poisson-like boundary) should fit."""
        X, y = tweedie_data
        # y must be strictly positive for Tweedie
        y = np.clip(y, 1e-6, None)
        model = DebiasedGLM(
            family="tweedie", tweedie_power=1.0, alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert model._is_fitted

    def test_tweedie_power_boundary_near_two(self, gamma_data):
        """Tweedie power=2.0 (Gamma boundary) should fit."""
        X, y = gamma_data
        model = DebiasedGLM(
            family="tweedie", tweedie_power=2.0, alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert model._is_fitted


# ---------------------------------------------------------------------------
# Internal helpers — debiased_glm
# ---------------------------------------------------------------------------


class TestCheckFamilyDebiased:
    def test_valid_families(self):
        for fam in ["poisson", "gamma", "tweedie"]:
            _check_family(fam, 1.5)  # no exception

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="family"):
            _check_family("binomial", 1.5)

    def test_tweedie_power_out_of_range(self):
        with pytest.raises(ValueError, match="tweedie_power"):
            _check_family("tweedie", 2.5)

    def test_tweedie_power_boundary_valid(self):
        _check_family("tweedie", 1.0)  # lower boundary
        _check_family("tweedie", 2.0)  # upper boundary


class TestHessianWeights:
    def test_poisson_equals_mu(self):
        mu = np.array([1.0, 2.0, 3.0])
        w = _hessian_weights(mu, "poisson", phi=1.0, p=1.0)
        np.testing.assert_array_equal(w, mu)

    def test_gamma_equals_one_over_phi(self):
        mu = np.array([1.0, 2.0, 3.0])
        phi = 2.0
        w = _hessian_weights(mu, "gamma", phi=phi, p=2.0)
        np.testing.assert_allclose(w, np.ones(3) / phi)

    def test_tweedie_formula(self):
        mu = np.array([2.0, 3.0])
        phi, p = 1.5, 1.5
        w = _hessian_weights(mu, "tweedie", phi=phi, p=p)
        expected = mu ** (2.0 - p) / phi
        np.testing.assert_allclose(w, expected)


class TestSqrtVariance:
    def test_poisson_sqrt_mu(self):
        mu = np.array([1.0, 4.0, 9.0])
        sv = _sqrt_variance(mu, "poisson", phi=1.0, p=1.0)
        np.testing.assert_allclose(sv, np.sqrt(mu))

    def test_gamma_formula(self):
        mu = np.array([2.0, 4.0])
        phi = 2.0
        sv = _sqrt_variance(mu, "gamma", phi=phi, p=2.0)
        expected = mu / np.sqrt(phi)
        np.testing.assert_allclose(sv, expected)

    def test_tweedie_formula(self):
        mu = np.array([2.0])
        phi, p = 1.5, 1.5
        sv = _sqrt_variance(mu, "tweedie", phi=phi, p=p)
        expected = mu ** (p / 2.0) * np.sqrt(phi)
        np.testing.assert_allclose(sv, expected)


class TestEstimateDispersion:
    def test_poisson_always_one(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 2.1, 3.1])
        phi = _estimate_dispersion(y, mu, "poisson", p=1.0, n_params=2)
        assert phi == 1.0

    def test_gamma_estimation(self):
        rng = np.random.default_rng(0)
        n = 1000
        mu = np.exp(rng.standard_normal(n) * 0.5)
        y = rng.gamma(3.0, mu / 3.0)
        phi = _estimate_dispersion(y, mu, "gamma", p=2.0, n_params=5)
        assert phi > 0
        # For Gamma with shape=3, phi ~ 1/3 ~ 0.33
        assert 0.1 < phi < 2.0


class TestVarianceFunction:
    def test_poisson_returns_mu(self):
        mu = np.array([1.0, 2.0])
        v = _variance_function(mu, "poisson", phi=1.0, p=1.0)
        np.testing.assert_array_equal(v, mu)

    def test_gamma_formula(self):
        mu = np.array([2.0, 3.0])
        phi = 2.0
        v = _variance_function(mu, "gamma", phi=phi, p=2.0)
        expected = mu ** 2 / phi
        np.testing.assert_allclose(v, expected)


# ---------------------------------------------------------------------------
# PenalizedGLMInference — additional edge cases
# ---------------------------------------------------------------------------


class TestPINBootstrapGamma:
    def test_gamma_bootstrap_ci_runs(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.bootstrap_ci(alpha=0.05, n_bootstrap=30, random_state=SEED)
        assert len(df) == P
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"], (
                f"Gamma bootstrap CI inverted for {row['feature']}"
            )

    def test_gamma_bootstrap_se_nan(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.bootstrap_ci(n_bootstrap=20, random_state=SEED)
        assert np.all(np.isnan(df["se"].values))

    def test_gamma_bootstrap_coef_equals_penalized(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.bootstrap_ci(n_bootstrap=20, random_state=SEED)
        np.testing.assert_allclose(
            df["coef"].values, df["coef_penalized"].values, atol=1e-12
        )


class TestPINBootstrapTweedie:
    def test_tweedie_bootstrap_ci_runs(self, tweedie_data):
        X, y = tweedie_data
        m = PenalizedGLMInference(
            family="tweedie", tweedie_power=1.5, alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.bootstrap_ci(alpha=0.05, n_bootstrap=25, random_state=SEED)
        assert len(df) == P
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]


class TestPINNestedCIs:
    def test_99_ci_wider_than_95(self, fitted_pin):
        """99% CIs should be at least as wide as 95% CIs."""
        df_95 = fitted_pin.confidence_intervals(alpha=0.05)
        df_99 = fitted_pin.confidence_intervals(alpha=0.01)
        w95 = df_95["ci_upper"] - df_95["ci_lower"]
        w99 = df_99["ci_upper"] - df_99["ci_lower"]
        assert np.all((w99 - w95).values >= -1e-10)

    def test_80_ci_narrower_than_95(self, fitted_pin):
        """80% CIs should be narrower than 95% CIs."""
        df_80 = fitted_pin.confidence_intervals(alpha=0.20)
        df_95 = fitted_pin.confidence_intervals(alpha=0.05)
        w80 = df_80["ci_upper"] - df_80["ci_lower"]
        w95 = df_95["ci_upper"] - df_95["ci_lower"]
        assert np.all((w95 - w80).values >= -1e-10)


class TestPINTwoFeature:
    def test_small_model_works(self):
        """Two-feature model should converge and produce valid output."""
        rng = np.random.default_rng(0)
        n, p = 500, 2
        X = rng.standard_normal((n, p))
        y = rng.poisson(np.exp(0.5 * X[:, 0]))
        m = PenalizedGLMInference(family="poisson", alpha=0.1, random_state=0).fit(X, y)
        df = m.confidence_intervals()
        assert len(df) == 2
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]


class TestPINLassoMode:
    def test_pure_lasso_fits(self, poisson_data):
        """l1_ratio=1.0 (Lasso) should produce sparse solution."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, l1_ratio=1.0, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        assert len(df) == P
        # Lasso should zero out some features
        assert m.selected_features_.sum() < P


class TestPINForestPlotAll:
    def test_forest_plot_all_features(self, fitted_pin):
        """forest_plot(only_selected=False) should not raise."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax_out = fitted_pin.forest_plot(only_selected=False, ax=ax)
        assert ax_out is ax
        plt.close("all")


class TestPINSummaryAndCIEquivalence:
    def test_summary_equals_confidence_intervals(self, fitted_pin):
        """summary() should be identical to confidence_intervals(alpha=0.05)."""
        df_sum = fitted_pin.summary()
        df_ci = fitted_pin.confidence_intervals(alpha=0.05)
        pd.testing.assert_frame_equal(df_sum, df_ci)

    def test_summary_custom_alpha(self, fitted_pin):
        df_01 = fitted_pin.summary(alpha=0.01)
        df_ci_01 = fitted_pin.confidence_intervals(alpha=0.01)
        pd.testing.assert_frame_equal(df_01, df_ci_01)


class TestPINGammaFamilyDetails:
    def test_gamma_phi_estimated_and_positive(self, gamma_data):
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert m.phi_ > 0
        assert m.phi_ != 1.0

    def test_gamma_ci_covers_true_coefs(self, gamma_data):
        """At least 1 of 3 true features should have CI covering true value."""
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="gamma", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        true_vals = {"x0": 0.4, "x1": 0.3, "x2": 0.2}
        covered = 0
        for name, val in true_vals.items():
            row = df[df["feature"] == name]
            if row.iloc[0]["selected"]:
                lo = row.iloc[0]["ci_lower"]
                hi = row.iloc[0]["ci_upper"]
                if lo <= val <= hi:
                    covered += 1
        assert covered >= 1, f"Expected >= 1 gamma coef covered, got {covered}"


class TestPINEdgeCases:
    def test_coef_penalized_consistent_with_selected(self, fitted_pin):
        """selected_features_ should match abs(coef_penalized_) > 1e-8."""
        expected = np.abs(fitted_pin.coef_penalized_) > 1e-8
        np.testing.assert_array_equal(fitted_pin.selected_features_, expected)

    def test_n_obs_and_n_features_correct(self, poisson_data, fitted_pin):
        X, _ = poisson_data
        assert fitted_pin.n_obs_ == N
        assert fitted_pin.n_features_ == P

    def test_lambda_positive(self, fitted_pin):
        assert fitted_pin.lambda_ > 0

    def test_bootstrap_with_explicit_random_state(self, fitted_pin):
        """Two calls with same random_state should give identical CI bounds."""
        df1 = fitted_pin.bootstrap_ci(n_bootstrap=30, random_state=77)
        df2 = fitted_pin.bootstrap_ci(n_bootstrap=30, random_state=77)
        np.testing.assert_allclose(
            df1["ci_lower"].values, df2["ci_lower"].values, atol=1e-12
        )

    def test_coef_penalized_col_in_asymptotic_df(self, fitted_pin):
        df = fitted_pin.confidence_intervals()
        assert "coef_penalized" in df.columns
        np.testing.assert_allclose(
            df["coef_penalized"].values,
            fitted_pin.coef_penalized_,
            atol=1e-12,
        )
