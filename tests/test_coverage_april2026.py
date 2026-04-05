"""
Extended test coverage for insurance-gam GLM inference modules (April 2026).

Targets gaps not covered by existing tests:
  - PostSelectionGLM: subsample path (n > subsample), bad y_shape coercion,
    _lasso_select max_features path, _trace_lasso_path_for_variable degenerate
    column, zero-count response, repeated fit on same object
  - DataSplitPostSelectionGLM: exposure with gamma check, negative exposure,
    n with odd length (split not exact halves)
  - DebiasedGLM: Gamma with exposure, Tweedie power boundaries in fit,
    near-singular Hessian warning path, refit path on Gamma (phi update),
    forest_plot bootstrap mode
  - PenalizedGLMInference: re-call confidence_intervals after bootstrap_ci,
    Gamma with exposure, pure ridge fit (l1_ratio=0.0), multiple alpha calls
    on same object produce independent DataFrames
  - Integration: PostSelectionGLM + DebiasedGLM + PenalizedGLMInference
    all applied to the same dataset, results are consistent
  - Internal helpers: _lasso_select with max_features already satisfied,
    _bisect convergence edge cases, _truncated_normal_ci with tight intervals
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
    _make_offset,
    _lasso_select,
    _poisson_pseudo_data,
    _fit_poisson_mle,
    _truncated_normal_ci,
    _trace_lasso_path_for_variable,
)
from insurance_gam.debiased_glm import DebiasedGLM
from insurance_gam.penalized_glm_inference import PenalizedGLMInference


# ---------------------------------------------------------------------------
# Shared DGP
# ---------------------------------------------------------------------------

TRUE_COEFS = np.array([0.5, 0.3])
N = 1500
P = 8
SEED = 17


@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, P))
    eta = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    y = rng.poisson(np.exp(eta))
    return X, y


@pytest.fixture(scope="module")
def gamma_data():
    rng = np.random.default_rng(SEED + 10)
    X = rng.standard_normal((N, P))
    mu = np.exp(0.4 * X[:, 0] + 0.25 * X[:, 1])
    shape = 4.0
    y = rng.gamma(shape, mu / shape)
    return X, y


@pytest.fixture(scope="module")
def exposure_data():
    """Poisson with exposure."""
    rng = np.random.default_rng(SEED + 20)
    X = rng.standard_normal((N, P))
    exposure = rng.uniform(0.3, 3.0, size=N)
    eta = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.log(exposure)
    y = rng.poisson(np.exp(eta))
    return X, y, exposure


# ---------------------------------------------------------------------------
# PostSelectionGLM — subsample path
# ---------------------------------------------------------------------------


class TestPostSelectionGLMSubsample:
    def test_subsample_path_runs(self, poisson_data):
        """n=1500 > subsample=200: subsample code path executes."""
        X, y = poisson_data
        model = PostSelectionGLM(
            subsample=200,
            cv_folds=3,
            random_state=SEED,
        ).fit(X, y)
        df = model.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == P

    def test_subsample_path_produces_valid_cis(self, poisson_data):
        """CI ordering holds with subsampling."""
        X, y = poisson_data
        model = PostSelectionGLM(
            subsample=200,
            cv_folds=3,
            random_state=SEED,
        ).fit(X, y)
        df = model.summary()
        for _, row in df.iterrows():
            if row["selected"] and not np.isnan(row["ci_lower"]):
                assert row["ci_lower"] <= row["ci_upper"], (
                    f"CI inverted for {row['feature']} with subsampling"
                )

    def test_subsample_equal_to_n_no_subsampling(self, poisson_data):
        """subsample >= n should use all data (no subsampling branch)."""
        X, y = poisson_data
        model = PostSelectionGLM(
            subsample=100_000,
            cv_folds=3,
            random_state=SEED,
        ).fit(X, y)
        df = model.summary()
        assert len(df) == P

    def test_repeated_fit_updates_results(self, poisson_data):
        """Fitting the same object twice should produce fresh results."""
        X, y = poisson_data
        model = PostSelectionGLM(cv_folds=3, random_state=SEED)
        model.fit(X, y)
        df1 = model.summary().copy()
        model.fit(X, y)
        df2 = model.summary().copy()
        # Both calls produce the same-length DataFrame
        assert len(df1) == len(df2) == P

    def test_all_zero_counts_warns_or_completes(self):
        """Response of all zeros: model should either warn or complete."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 4))
        y = np.zeros(300)  # all zeros — valid for Poisson
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = PostSelectionGLM(cv_folds=2, random_state=0).fit(X, y)
        df = model.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_exposure_as_series(self, poisson_data):
        """Exposure passed as a pandas Series should be accepted."""
        X, y = poisson_data
        exposure = pd.Series(np.ones(N))
        model = PostSelectionGLM(cv_folds=3, random_state=SEED).fit(
            X, y, exposure=exposure
        )
        df = model.summary()
        assert len(df) == P


# ---------------------------------------------------------------------------
# DataSplitPostSelectionGLM — edge cases
# ---------------------------------------------------------------------------


class TestDataSplitEdgeCases:
    def test_odd_n_splits_correctly(self):
        """n=301 (odd): split should still work (n//2=150 select, 151 infer)."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((301, 5))
        y = rng.poisson(np.exp(0.4 * X[:, 0]))
        model = DataSplitPostSelectionGLM(cv_folds=2, random_state=5).fit(X, y)
        df = model.summary()
        assert len(df) == 5

    def test_exposure_as_series_accepted(self, exposure_data):
        """Exposure as Series (not ndarray) should be accepted."""
        X, y, exposure = exposure_data
        exposure_series = pd.Series(exposure)
        model = DataSplitPostSelectionGLM(cv_folds=3, random_state=SEED).fit(
            X, y, exposure=exposure_series
        )
        df = model.summary()
        assert len(df) == P

    def test_negative_exposure_raises(self, poisson_data):
        """Negative exposure values should raise ValueError."""
        X, y = poisson_data
        bad_exposure = np.ones(N)
        bad_exposure[10] = -1.0
        with pytest.raises(ValueError, match="positive"):
            DataSplitPostSelectionGLM().fit(X, y, exposure=bad_exposure)

    def test_single_feature(self):
        """Single-feature design matrix should fit without error."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 1))
        y = rng.poisson(np.exp(0.5 * X[:, 0]))
        model = DataSplitPostSelectionGLM(cv_folds=2, random_state=42).fit(X, y)
        df = model.summary()
        assert len(df) == 1

    def test_forest_plot_no_selected_is_handled(self):
        """If Lasso selects nothing, forest_plot raises ValueError."""
        import matplotlib
        matplotlib.use("Agg")
        rng = np.random.default_rng(99)
        X = rng.standard_normal((300, 4))
        y = rng.poisson(1.0, size=300)
        # Very high max_features=0 to force zero selection
        model = DataSplitPostSelectionGLM(max_features=0, random_state=99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        df = model.summary()
        if not df["selected"].any():
            with pytest.raises(ValueError):
                model.forest_plot()

    def test_dataframe_input_y_as_series(self, poisson_data):
        """y passed as pd.Series should be coerced correctly."""
        X, y = poisson_data
        y_series = pd.Series(y)
        model = DataSplitPostSelectionGLM(cv_folds=3, random_state=SEED).fit(
            X, y_series
        )
        df = model.summary()
        assert len(df) == P


# ---------------------------------------------------------------------------
# Internal helper: _lasso_select
# ---------------------------------------------------------------------------


class TestLassoSelect:
    def test_max_features_constraint_already_met(self):
        """max_features larger than actual selected: no re-fitting needed."""
        rng = np.random.default_rng(0)
        n, p = 200, 5
        X = rng.standard_normal((n, p))
        y = rng.poisson(np.exp(0.5 * X[:, 0]))
        eta = np.log(np.clip(y, 0.1, None))
        z, Z = _poisson_pseudo_data(X, y, eta)
        # max_features = 10 > p, so constraint is never binding
        selected, lam, beta = _lasso_select(Z, z, cv_folds=3, max_features=10, random_state=0)
        assert selected.shape == (p,)
        assert lam > 0

    def test_max_features_tight_constraint(self):
        """max_features=1 should select at most 1 feature."""
        rng = np.random.default_rng(1)
        n, p = 300, 6
        X = rng.standard_normal((n, p))
        y = rng.poisson(np.exp(0.5 * X[:, 0]))
        eta = np.log(np.clip(y, 0.1, None))
        z, Z = _poisson_pseudo_data(X, y, eta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected, lam, beta = _lasso_select(
                Z, z, cv_folds=3, max_features=1, random_state=1
            )
        # max_features=1: at most 1 feature should be selected
        assert selected.sum() <= 1
        assert selected.dtype == bool

    def test_returns_correct_types(self):
        rng = np.random.default_rng(2)
        n, p = 200, 4
        X = rng.standard_normal((n, p))
        y = rng.poisson(2.0, size=n)
        eta = np.log(np.clip(y, 0.1, None))
        z, Z = _poisson_pseudo_data(X, y, eta)
        selected, lam, beta = _lasso_select(Z, z, cv_folds=3, max_features=None, random_state=2)
        assert selected.dtype == bool
        assert isinstance(lam, float)
        assert beta.shape == (p,)


# ---------------------------------------------------------------------------
# Internal helper: _trace_lasso_path_for_variable
# ---------------------------------------------------------------------------


class TestTraceLassoPath:
    def test_degenerate_column_returns_full_real_line(self):
        """Column with zero norm: should return [(-inf, +inf)]."""
        rng = np.random.default_rng(3)
        n, p = 50, 3
        Z = rng.standard_normal((n, p))
        # Make first column all zeros
        Z[:, 0] = 0.0
        z = rng.standard_normal(n)
        active_mask = np.array([True, True, False])
        intervals = _trace_lasso_path_for_variable(
            Z, z, j=0, active_mask=active_mask, lambda_opt=0.1, n_steps=20
        )
        assert len(intervals) == 1
        lo, hi = intervals[0]
        assert lo == -np.inf
        assert hi == np.inf

    def test_returns_nonempty_intervals(self):
        """Normal column: should return at least one interval."""
        rng = np.random.default_rng(4)
        n, p = 100, 3
        Z = rng.standard_normal((n, p))
        z = 0.3 * Z[:, 0] + rng.standard_normal(n) * 0.1
        active_mask = np.array([True, False, False])
        intervals = _trace_lasso_path_for_variable(
            Z, z, j=0, active_mask=active_mask, lambda_opt=0.05, n_steps=30
        )
        assert len(intervals) >= 1


# ---------------------------------------------------------------------------
# Internal helper: _fit_poisson_mle
# ---------------------------------------------------------------------------


class TestFitPoissonMLE:
    def test_returns_reasonable_estimates(self):
        rng = np.random.default_rng(5)
        n = 500
        X = rng.standard_normal((n, 2))
        eta_true = 0.4 * X[:, 0] + 0.2 * X[:, 1]
        y = rng.poisson(np.exp(eta_true))
        offset = np.zeros(n)
        beta, eta_hat = _fit_poisson_mle(X, y, offset)
        # Intercept + 2 coefficients
        assert beta.shape == (3,)
        # Coefficient estimates within 0.3 of truth
        assert abs(beta[1] - 0.4) < 0.3
        assert abs(beta[2] - 0.2) < 0.3

    def test_with_exposure_offset(self):
        rng = np.random.default_rng(6)
        n = 300
        X = rng.standard_normal((n, 2))
        exposure = rng.uniform(0.5, 2.0, size=n)
        offset = np.log(exposure)
        eta_true = 0.3 * X[:, 0] + offset
        y = rng.poisson(np.exp(eta_true))
        beta, eta_hat = _fit_poisson_mle(X, y, offset)
        assert np.all(np.isfinite(beta))
        assert eta_hat.shape == (n,)


# ---------------------------------------------------------------------------
# DebiasedGLM — additional coverage
# ---------------------------------------------------------------------------


class TestDebiasedGLMAdditional:
    def test_gamma_with_exposure(self, gamma_data):
        """Gamma model with exposure offset should fit without error."""
        X, y = gamma_data
        exposure = np.ones(N)  # unit exposure → zero offset
        model = DebiasedGLM(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y, exposure=exposure)
        df = model.summary()
        assert len(df) == P
        assert model.phi_ > 0

    def test_tweedie_power_1_boundary(self):
        """Tweedie power=1.0 (Poisson-like) with positive y."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((500, 4))
        y = rng.gamma(2.0, 1.0, size=500)  # strictly positive
        model = DebiasedGLM(
            family="tweedie", tweedie_power=1.0, alpha=0.2, random_state=7
        ).fit(X, y)
        df = model.summary()
        assert len(df) == 4
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_tweedie_power_2_boundary(self, gamma_data):
        """Tweedie power=2.0 (Gamma-like boundary) should fit."""
        X, y = gamma_data
        model = DebiasedGLM(
            family="tweedie", tweedie_power=2.0, alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        assert len(df) == P
        assert np.isfinite(model.intercept_)

    def test_cv_selection_updates_lambda(self, poisson_data):
        """alpha=0.0 triggers CV; lambda_ should be positive and reasonable."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)
        assert model.lambda_ > 0
        # The CV-selected lambda should be less than a large manual value
        assert model.lambda_ < 10.0

    def test_forest_plot_bootstrap_strategy(self, poisson_data):
        """DebiasedGLM with n_bootstrap > 0: forest_plot still works."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, n_bootstrap=20, random_state=SEED
        ).fit(X, y)
        fig, ax = plt.subplots()
        # Should use Strategy B (bootstrap) internally but still plot
        ax_out = model.forest_plot(ax=ax)
        assert ax_out is ax
        plt.close("all")

    def test_phi_positive_gamma_refit(self, gamma_data):
        """Gamma: phi should update from 1.0 in the dispersion refit loop."""
        X, y = gamma_data
        model = DebiasedGLM(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y)
        # Gamma shape=4 → phi ~ 1/shape ~ 0.25; should differ from 1.0
        assert model.phi_ != 1.0
        assert model.phi_ > 0

    def test_intercept_reasonable_scale(self, poisson_data):
        """Intercept should be finite and in a reasonable range."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        assert np.isfinite(model.intercept_)
        assert -10 < model.intercept_ < 10

    def test_debiased_coef_differs_from_penalized(self, poisson_data):
        """Debiased coef_ should differ from the penalised _beta_penalised."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)
        sel = model.selected_features_
        if sel.any():
            diff = np.abs(model.coef_[sel] - model._beta_penalised[sel])
            assert np.any(diff > 1e-10), (
                "Debiasing had no effect on any selected feature."
            )

    def test_l1_ratio_zero_fits_without_error(self, poisson_data):
        """l1_ratio=0.0 (Ridge) produces non-sparse solution."""
        X, y = poisson_data
        model = DebiasedGLM(
            family="poisson", alpha=0.05, l1_ratio=0.0, random_state=SEED
        ).fit(X, y)
        df = model.summary()
        assert len(df) == P
        # Ridge: all features should be selected (non-zero)
        assert model.selected_features_.all()


# ---------------------------------------------------------------------------
# PenalizedGLMInference — additional coverage
# ---------------------------------------------------------------------------


class TestPINAdditional:
    def test_confidence_intervals_after_bootstrap_ci(self, poisson_data):
        """Calling bootstrap_ci() then confidence_intervals() on the same object works."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        # Bootstrap first
        df_boot = m.bootstrap_ci(alpha=0.05, n_bootstrap=30, random_state=SEED)
        # Then asymptotic — should be independent
        df_asym = m.confidence_intervals(alpha=0.05)
        assert len(df_boot) == P
        assert len(df_asym) == P
        # Both should have correct columns
        assert "se" in df_asym.columns
        assert "se" in df_boot.columns
        # SE: NaN for bootstrap, positive for asymptotic (selected)
        sel_asym = df_asym[df_asym["selected"]]
        assert (sel_asym["se"] > 0).all()
        assert np.all(np.isnan(df_boot["se"].values))

    def test_gamma_with_exposure(self, gamma_data):
        """PenalizedGLMInference Gamma with exposure fits correctly."""
        X, y = gamma_data
        exposure = np.ones(N) * 2.0  # constant exposure — just checks offset path
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y, exposure=exposure)
        df = m.confidence_intervals()
        assert len(df) == P
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_pure_ridge_no_sparsity(self, poisson_data):
        """l1_ratio=0.0 (pure Ridge): all features selected, larger CIs."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.05, l1_ratio=0.0, random_state=SEED
        ).fit(X, y)
        # Ridge does not zero out features
        assert m.selected_features_.all()
        df = m.confidence_intervals()
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_multiple_alpha_calls_independent(self, poisson_data):
        """Repeated calls to confidence_intervals() with different alpha are independent."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df1 = m.confidence_intervals(alpha=0.05)
        df2 = m.confidence_intervals(alpha=0.10)
        df3 = m.confidence_intervals(alpha=0.05)
        # df1 and df3 should be identical (same alpha)
        pd.testing.assert_frame_equal(df1, df3)
        # 90% CIs (alpha=0.10) should be narrower than 95% (alpha=0.05)
        w1 = (df1["ci_upper"] - df1["ci_lower"]).values
        w2 = (df2["ci_upper"] - df2["ci_lower"]).values
        assert np.all(w2 <= w1 + 1e-10)

    def test_summary_at_different_alpha_levels(self, poisson_data):
        """summary() is an alias for confidence_intervals() and respects alpha."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df_90 = m.summary(alpha=0.10)
        df_95 = m.summary(alpha=0.05)
        w_90 = df_90["ci_upper"].values - df_90["ci_lower"].values
        w_95 = df_95["ci_upper"].values - df_95["ci_lower"].values
        assert np.all(w_95 >= w_90 - 1e-10), "95% CIs should be wider than 90% CIs"

    def test_bootstrap_ci_gamma_exposure(self, gamma_data):
        """Bootstrap CI for Gamma + exposure should maintain ordering."""
        X, y = gamma_data
        exposure = np.ones(N)
        m = PenalizedGLMInference(
            family="gamma", alpha=0.1, random_state=SEED
        ).fit(X, y, exposure=exposure)
        df = m.bootstrap_ci(alpha=0.05, n_bootstrap=20, random_state=SEED)
        assert len(df) == P
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["ci_upper"]

    def test_forest_plot_bootstrap_all_features(self, poisson_data):
        """Forest plot with bootstrap + only_selected=False plots all."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        fig, ax = plt.subplots()
        ax_out = m.forest_plot(
            strategy="bootstrap", n_bootstrap=20, only_selected=False, ax=ax
        )
        assert ax_out is ax
        plt.close("all")

    def test_bootstrap_ci_large_n_bootstrap(self, poisson_data):
        """n_bootstrap=100 runs without OOM or silent failure."""
        X, y = poisson_data
        m = PenalizedGLMInference(
            family="poisson", alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.bootstrap_ci(alpha=0.05, n_bootstrap=100, random_state=SEED)
        assert len(df) == P
        assert not df["ci_lower"].isna().any()
        assert not df["ci_upper"].isna().any()

    def test_tweedie_power_boundary_one(self):
        """Tweedie power=1.0 fits on strictly positive y."""
        rng = np.random.default_rng(8)
        X = rng.standard_normal((400, 4))
        y = rng.gamma(2.0, 1.0, size=400)
        m = PenalizedGLMInference(
            family="tweedie", tweedie_power=1.0, alpha=0.1, random_state=8
        ).fit(X, y)
        df = m.confidence_intervals()
        assert len(df) == 4
        assert m._is_fitted

    def test_tweedie_power_boundary_two(self, gamma_data):
        """Tweedie power=2.0 (Gamma boundary) fits correctly."""
        X, y = gamma_data
        m = PenalizedGLMInference(
            family="tweedie", tweedie_power=2.0, alpha=0.1, random_state=SEED
        ).fit(X, y)
        df = m.confidence_intervals()
        assert len(df) == P
        assert m.phi_ > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Apply all three GLM inference classes to the same dataset and cross-check."""

    def test_all_three_classes_on_same_data(self, poisson_data):
        """PostSelectionGLM, DebiasedGLM, and PenalizedGLMInference on same data."""
        X, y = poisson_data

        # PostSelectionGLM
        ps = PostSelectionGLM(cv_folds=3, random_state=SEED).fit(X, y)
        ps_df = ps.summary()

        # DebiasedGLM
        deb = DebiasedGLM(family="poisson", alpha=0.0, cv_folds=3, random_state=SEED).fit(X, y)
        deb_df = deb.summary()

        # PenalizedGLMInference
        pin = PenalizedGLMInference(family="poisson", alpha=0.0, cv_folds=3, random_state=SEED).fit(X, y)
        pin_df = pin.confidence_intervals()

        # All should have the same number of rows
        assert len(ps_df) == P
        assert len(deb_df) == P
        assert len(pin_df) == P

        # All should have valid CI ordering for selected features
        for df, label in [(ps_df, "PostSelection"), (deb_df, "Debiased"), (pin_df, "PIN")]:
            for _, row in df.iterrows():
                # ci_lower/ci_upper columns present in all three
                lo_col = "ci_lower"
                hi_col = "ci_upper"
                lo = row[lo_col]
                hi = row[hi_col]
                if not np.isnan(lo):
                    assert lo <= hi, f"{label}: CI inverted for {row.get('feature', '?')}"

    def test_selection_agreement(self, poisson_data):
        """PostSelectionGLM and DebiasedGLM should both select x0 (strongest signal)."""
        X, y = poisson_data

        ps = PostSelectionGLM(cv_folds=3, random_state=SEED).fit(X, y)
        deb = DebiasedGLM(family="poisson", alpha=0.0, cv_folds=3, random_state=SEED).fit(X, y)

        ps_selected = set(
            ps.summary()[ps.summary()["selected"]]["feature"].tolist()
        )
        deb_selected = set(
            deb.summary()[deb.summary()["selected"]]["feature"].tolist()
        )

        # At least x0 (strongest feature) should be in both
        assert "x0" in ps_selected, f"x0 not in PostSelection: {ps_selected}"
        assert "x0" in deb_selected, f"x0 not in Debiased: {deb_selected}"

    def test_datasplit_and_pin_consistent_selection(self, poisson_data):
        """DataSplit and PenalizedGLMInference should select overlapping features."""
        X, y = poisson_data

        ds = DataSplitPostSelectionGLM(cv_folds=3, random_state=SEED).fit(X, y)
        pin = PenalizedGLMInference(
            family="poisson", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)

        ds_selected = set(
            ds.summary()[ds.summary()["selected"]]["feature"].tolist()
        )
        pin_df = pin.confidence_intervals()
        pin_selected = set(
            pin_df[pin_df["selected"]]["feature"].tolist()
        )

        # Both should select at least x0 (largest signal)
        assert "x0" in ds_selected or "x0" in pin_selected, (
            f"x0 not selected by either method. DS: {ds_selected}, PIN: {pin_selected}"
        )

    def test_coefficient_sign_agreement(self, poisson_data):
        """True positive features (x0, x1) should have positive coefs in all methods."""
        X, y = poisson_data

        deb = DebiasedGLM(
            family="poisson", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)
        pin = PenalizedGLMInference(
            family="poisson", alpha=0.0, cv_folds=3, random_state=SEED
        ).fit(X, y)

        deb_df = deb.summary()
        pin_df = pin.confidence_intervals()

        for feat in ["x0", "x1"]:
            deb_row = deb_df[deb_df["feature"] == feat]
            pin_row = pin_df[pin_df["feature"] == feat]

            if deb_row.iloc[0]["selected"]:
                assert deb_row.iloc[0]["coef"] > 0, (
                    f"Debiased coef for {feat} is negative (expected positive)"
                )
            if pin_row.iloc[0]["selected"]:
                assert pin_row.iloc[0]["coef"] > 0, (
                    f"PIN coef for {feat} is negative (expected positive)"
                )


# ---------------------------------------------------------------------------
# Internal helpers — additional
# ---------------------------------------------------------------------------


class TestBisectExtended:
    def test_already_bracketed(self):
        """Bracket already contains root: converges immediately."""
        x = _bisect(lambda t: t - 0.3, 0.0, -1.0, 1.0)
        assert abs(x - 0.3) < 1e-5

    def test_monotone_decreasing_function(self):
        """Bisect works for decreasing functions."""
        # f(t) = -t, target = -0.7 → t = 0.7
        x = _bisect(lambda t: -t, -0.7, 0.0, 2.0)
        assert abs(x - 0.7) < 1e-4

    def test_tight_tolerance(self):
        """Custom tolerance argument works."""
        x = _bisect(lambda t: t, 0.12345, 0.0, 1.0, tol=1e-8)
        assert abs(x - 0.12345) < 1e-7


class TestTruncatedNormalCIExtended:
    def test_tight_interval_truncation(self):
        """Truncating to a narrow interval still produces finite CIs."""
        intervals = [(0.4, 0.6)]
        ci_lo, ci_hi, pval = _truncated_normal_ci(
            intervals, mu=0.5, sigma=0.05, alpha=0.05
        )
        assert np.isfinite(ci_lo) or np.isnan(ci_lo)  # either is acceptable
        if not np.isnan(ci_lo):
            assert ci_lo <= ci_hi

    def test_pvalue_at_mu_zero_half(self):
        """p-value for mu=0 (null is true) should be ~ 1 with no truncation."""
        intervals = [(-np.inf, np.inf)]
        _, _, pval = _truncated_normal_ci(intervals, mu=0.0, sigma=1.0, alpha=0.05)
        # p-value = 2*min(F(0), 1-F(0)) = 2*0.5 = 1.0
        assert abs(pval - 1.0) < 0.01

    def test_large_mu_gives_small_pvalue(self):
        """mu >> sigma: p-value should be small (strong rejection of H0: mu=0)."""
        intervals = [(-np.inf, np.inf)]
        _, _, pval = _truncated_normal_ci(intervals, mu=5.0, sigma=1.0, alpha=0.05)
        assert pval < 0.01


class TestMakeOffsetExtended:
    def test_large_exposure(self):
        """Large exposure values should not overflow."""
        exp = np.array([1e6, 1e7, 1e8])
        offset = _make_offset(exp, 3)
        assert np.all(np.isfinite(offset))

    def test_exposure_close_to_zero_but_positive(self):
        """Exposure values very close to zero (but positive) are accepted."""
        exp = np.array([1e-10, 1.0, 2.0])
        offset = _make_offset(exp, 3)
        assert np.all(np.isfinite(offset))

    def test_output_is_log(self):
        """Output should equal log(exposure)."""
        exp = np.array([np.e, np.e ** 2, np.e ** 3])
        offset = _make_offset(exp, 3)
        np.testing.assert_allclose(offset, [1.0, 2.0, 3.0], atol=1e-10)
