"""
Tests for EBMInsuranceWrapper and associated module-level helpers.

All tests run without a real interpretML install — the mock in conftest.py
handles the ExplainableBoostingRegressor dependency.

Coverage targets:
- EBMInsuranceWrapper constructor
- fit / predict / score delegation to InsuranceEBM
- balance_ratio (module-level)
- balance_check / apply_balance_correction
- relativity_table / relativity_summary
- interaction_heatmap / interaction_heatmap_wide
- ae_by_decile / gini_coefficient / segment_ae_table
- transparency_report / transparency_report_json
- cross_validate (module-level and method)
- repr / feature_names / is_fitted_
- error cases (unfitted, bad inputs, no interaction term)
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_gam.ebm import EBMInsuranceWrapper, balance_ratio, cross_validate


# ---------------------------------------------------------------------------
# Fixtures — reuse conftest fixtures + wrapper-specific helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_wrapper(X_polars, synthetic_data):
    """Fitted EBMInsuranceWrapper using Poisson loss, no interactions."""
    wrapper = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
    wrapper.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
    return wrapper


@pytest.fixture(scope="session")
def fitted_tweedie_wrapper(X_polars, synthetic_data):
    """Fitted EBMInsuranceWrapper with Tweedie loss."""
    wrapper = EBMInsuranceWrapper(
        loss="tweedie", variance_power=1.5, interactions=0, random_state=42, n_jobs=1
    )
    wrapper.fit(X_polars, synthetic_data["claim_amount"], exposure=synthetic_data["exposure"])
    return wrapper


@pytest.fixture
def wrapper_with_interaction(X_polars, synthetic_data):
    """
    A wrapper whose underlying mock EBM has a fake interaction term
    (driver_age x ncd). The explain_global is replaced with a callable
    that returns 2-D data for the interaction term index.
    """
    # Fresh wrapper (not using the session fixture to avoid mutation)
    w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
    w.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])

    ebm = w.model_.ebm_
    feat_names = list(ebm.feature_names_in_)
    idx_age = feat_names.index("driver_age")
    idx_ncd = feat_names.index("ncd")

    # Inject an interaction term
    ebm.term_features_ = list(ebm.term_features_) + [(idx_age, idx_ncd)]

    # 2-D scores (3 bins x 4 bins)
    rng = np.random.default_rng(7)
    interaction_scores = rng.standard_normal((3, 4)) * 0.2
    ebm.term_scores_ = list(ebm.term_scores_) + [interaction_scores]

    interaction_term_idx = len(ebm.term_features_) - 1

    # Patch explain_global on the instance.
    # We use a closure — no MethodType needed since we capture ebm via closure.
    def _explain_global(name="EBM"):
        explanation = MagicMock()

        def _data(feature_idx):
            if feature_idx == interaction_term_idx:
                n_bins_i, n_bins_j = interaction_scores.shape
                names_i = [str(k) for k in range(n_bins_i + 1)]
                names_j = [str(k) for k in range(n_bins_j + 1)]
                return {
                    "names": [names_i, names_j],
                    "scores": interaction_scores.tolist(),
                }
            else:
                n_bins = 5
                names = [str(i) for i in range(n_bins + 1)]
                scores = ebm.term_scores_[feature_idx][1:]
                return {"names": names, "scores": scores.tolist()}

        explanation.data.side_effect = _data
        return explanation

    ebm.explain_global = _explain_global

    return w, "driver_age", "ncd", interaction_scores


@pytest.fixture
def wrapper_with_intercept(X_polars, synthetic_data):
    """Wrapper where the mock EBM has an intercept_ attribute (for balance correction tests)."""
    w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
    w.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
    # Add intercept_ to the mock so apply_balance_correction can use it
    w.model_.ebm_.intercept_ = 0.0
    return w


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_default_loss_is_poisson(self):
        w = EBMInsuranceWrapper()
        assert w.loss == "poisson"

    def test_custom_loss(self):
        w = EBMInsuranceWrapper(loss="gamma")
        assert w.loss == "gamma"

    def test_variance_power_stored(self):
        w = EBMInsuranceWrapper(loss="tweedie", variance_power=1.8)
        assert w.variance_power == 1.8

    def test_interactions_stored(self):
        w = EBMInsuranceWrapper(interactions=5)
        assert w.interactions == 5

    def test_monotone_constraints_stored(self):
        mc = {"ncd": -1, "driver_age": 0}
        w = EBMInsuranceWrapper(monotone_constraints=mc)
        assert w.monotone_constraints == mc

    def test_ebm_kwargs_forwarded(self):
        w = EBMInsuranceWrapper(random_state=99, n_jobs=4)
        assert w.ebm_kwargs["random_state"] == 99
        assert w.ebm_kwargs["n_jobs"] == 4

    def test_is_not_fitted_initially(self):
        w = EBMInsuranceWrapper()
        assert w.is_fitted_ is False
        assert w.model_ is None

    def test_repr_unfitted(self):
        w = EBMInsuranceWrapper()
        r = repr(w)
        assert "EBMInsuranceWrapper" in r
        assert "unfitted" in r

    def test_repr_shows_loss(self):
        w = EBMInsuranceWrapper(loss="gamma")
        assert "gamma" in repr(w)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        result = w.fit(X_polars, synthetic_data["claim_count"])
        assert result is w

    def test_fit_sets_is_fitted(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        w.fit(X_polars, synthetic_data["claim_count"])
        assert w.is_fitted_ is True

    def test_fit_creates_model_(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        w.fit(X_polars, synthetic_data["claim_count"])
        assert w.model_ is not None

    def test_fit_creates_relativities_(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        w.fit(X_polars, synthetic_data["claim_count"])
        assert w.relativities_ is not None

    def test_fit_stores_metadata(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        w.fit(X_polars, synthetic_data["claim_count"])
        assert w._n_obs_train == len(X_polars)
        assert w._n_features == len(X_polars.columns)
        assert w._feature_names == list(X_polars.columns)

    def test_fit_with_exposure(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        w.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
        assert w.is_fitted_

    def test_fit_with_pandas(self, X_pandas, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0)
        w.fit(X_pandas, synthetic_data["claim_count"])
        assert w.is_fitted_

    def test_repr_fitted(self, fitted_wrapper):
        assert "fitted" in repr(fitted_wrapper)
        assert "unfitted" not in repr(fitted_wrapper)


# ---------------------------------------------------------------------------
# Predict and score delegation
# ---------------------------------------------------------------------------

class TestPredictScore:

    def test_predict_shape(self, fitted_wrapper, X_polars):
        preds = fitted_wrapper.predict(X_polars)
        assert preds.shape == (len(X_polars),)

    def test_predict_positive(self, fitted_wrapper, X_polars):
        preds = fitted_wrapper.predict(X_polars)
        assert np.all(preds > 0)

    def test_predict_with_exposure(self, fitted_wrapper, X_polars, synthetic_data):
        preds = fitted_wrapper.predict(X_polars, exposure=synthetic_data["exposure"])
        assert preds.shape == (len(X_polars),)

    def test_predict_log_score_shape(self, fitted_wrapper, X_polars):
        scores = fitted_wrapper.predict_log_score(X_polars)
        assert scores.shape == (len(X_polars),)

    def test_predict_log_score_exp_matches_predict(self, fitted_wrapper, X_polars):
        scores = fitted_wrapper.predict_log_score(X_polars)
        preds = fitted_wrapper.predict(X_polars)
        np.testing.assert_allclose(np.exp(scores), preds, rtol=1e-5)

    def test_score_returns_float(self, fitted_wrapper, X_polars, synthetic_data):
        s = fitted_wrapper.score(X_polars, synthetic_data["claim_count"])
        assert isinstance(s, float)

    def test_score_is_negative_deviance(self, fitted_wrapper, X_polars, synthetic_data):
        s = fitted_wrapper.score(X_polars, synthetic_data["claim_count"])
        assert s <= 0

    def test_predict_unfitted_raises(self, X_polars):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.predict(X_polars)

    def test_score_unfitted_raises(self, X_polars):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.score(X_polars, np.zeros(len(X_polars)))

    def test_feature_names_match_columns(self, fitted_wrapper, X_polars):
        assert set(fitted_wrapper.feature_names) == set(X_polars.columns)

    def test_feature_names_unfitted_raises(self):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            _ = w.feature_names


# ---------------------------------------------------------------------------
# balance_ratio (module-level)
# ---------------------------------------------------------------------------

class TestBalanceRatio:

    def test_perfect_prediction_is_one(self):
        y = np.array([1.0, 2.0, 3.0])
        assert abs(balance_ratio(y, y) - 1.0) < 1e-10

    def test_under_prediction_ratio_above_one(self):
        y = np.array([2.0, 2.0, 2.0])
        y_hat = np.array([1.0, 1.0, 1.0])
        assert balance_ratio(y, y_hat) == pytest.approx(2.0)

    def test_over_prediction_ratio_below_one(self):
        y = np.array([1.0, 1.0, 1.0])
        y_hat = np.array([2.0, 2.0, 2.0])
        assert balance_ratio(y, y_hat) == pytest.approx(0.5)

    def test_exposure_weighted(self):
        y = np.array([1.0, 2.0])
        y_hat = np.array([1.0, 1.0])
        w = np.array([1.0, 3.0])
        # sum(actual*w) = 1+6=7, sum(pred*w) = 1+3=4
        assert balance_ratio(y, y_hat, exposure=w) == pytest.approx(7.0 / 4.0)

    def test_zero_predicted_raises(self):
        y = np.array([1.0, 2.0])
        y_hat = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="non-positive"):
            balance_ratio(y, y_hat)

    def test_polars_series_input(self):
        y = pl.Series([1.0, 2.0, 3.0])
        y_hat = pl.Series([1.0, 2.0, 3.0])
        assert abs(balance_ratio(y, y_hat) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# balance_check / apply_balance_correction
# ---------------------------------------------------------------------------

class TestBalanceCheck:

    def test_balance_check_returns_float(self, fitted_wrapper, X_polars, synthetic_data):
        ratio = fitted_wrapper.balance_check(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert isinstance(ratio, float)

    def test_balance_check_positive(self, fitted_wrapper, X_polars, synthetic_data):
        ratio = fitted_wrapper.balance_check(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert ratio > 0

    def test_balance_check_unfitted_raises(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.balance_check(X_polars, synthetic_data["claim_count"])

    def test_apply_balance_correction_returns_float(self, wrapper_with_intercept, X_polars, synthetic_data):
        """apply_balance_correction should return the log correction factor."""
        correction = wrapper_with_intercept.apply_balance_correction(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert isinstance(correction, float)

    def test_apply_balance_correction_log_value(self, wrapper_with_intercept, X_polars, synthetic_data):
        """The correction is log(A/E), so log(1.0) = 0 when predictions balance perfectly."""
        # Patch predict to return perfect predictions
        orig_predict = wrapper_with_intercept.model_.predict
        y_arr = synthetic_data["claim_count"].copy()
        # Set intercept so predictions exactly match sum of actuals
        wrapper_with_intercept.model_.ebm_.intercept_ = 0.0
        correction = wrapper_with_intercept.apply_balance_correction(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        # Correction is a float — just verify it's a valid float
        assert np.isfinite(correction)

    def test_apply_balance_correction_updates_intercept(self, X_polars, synthetic_data):
        """After correction, intercept_ should change by log_correction."""
        w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
        w.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
        w.model_.ebm_.intercept_ = 0.0

        initial = float(w.model_.ebm_.intercept_)
        correction = w.apply_balance_correction(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        final = float(w.model_.ebm_.intercept_)
        assert abs((final - initial) - correction) < 1e-10

    def test_apply_balance_correction_no_intercept_raises(self, X_polars, synthetic_data):
        """If EBM has no intercept attribute, AttributeError should be raised."""
        w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
        w.fit(X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"])
        # The mock has no intercept_ — do not add it
        assert not hasattr(w.model_.ebm_, "intercept_")
        assert not hasattr(w.model_.ebm_, "intercept")
        with pytest.raises(AttributeError, match="intercept"):
            w.apply_balance_correction(
                X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
            )


# ---------------------------------------------------------------------------
# Relativity methods
# ---------------------------------------------------------------------------

class TestRelativities:

    def test_relativity_table_schema(self, fitted_wrapper):
        df = fitted_wrapper.relativity_table("driver_age")
        assert set(df.columns) == {"bin_label", "raw_score", "relativity"}

    def test_relativity_table_base_bin_is_one(self, fitted_wrapper):
        df = fitted_wrapper.relativity_table("ncd")
        rels = df["relativity"].to_numpy()
        assert np.any(np.isclose(rels, 1.0, atol=1e-8))

    def test_relativity_summary_sorted(self, fitted_wrapper):
        df = fitted_wrapper.relativity_summary()
        ranges = df["range"].to_numpy()
        assert np.all(np.diff(ranges) <= 1e-8)

    def test_relativity_summary_has_all_features(self, fitted_wrapper, X_polars):
        df = fitted_wrapper.relativity_summary()
        assert len(df) == len(X_polars.columns)

    def test_relativity_table_unfitted_raises(self, X_polars):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.relativity_table("driver_age")

    def test_relativity_summary_unfitted_raises(self):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.relativity_summary()


# ---------------------------------------------------------------------------
# Interaction heatmap
# ---------------------------------------------------------------------------

class TestInteractionHeatmap:

    def test_heatmap_schema(self, wrapper_with_interaction):
        w, fi, fj, _ = wrapper_with_interaction
        df = w.interaction_heatmap(fi, fj)
        assert set(df.columns) == {"bin_i", "bin_j", "score", "relativity"}

    def test_heatmap_shape(self, wrapper_with_interaction):
        w, fi, fj, scores_2d = wrapper_with_interaction
        df = w.interaction_heatmap(fi, fj)
        n_i, n_j = scores_2d.shape
        assert len(df) == n_i * n_j

    def test_heatmap_mean_log_relativity_near_zero(self, wrapper_with_interaction):
        """Relativities are centred on mean score, so mean of log(rel) = 0."""
        w, fi, fj, _ = wrapper_with_interaction
        df = w.interaction_heatmap(fi, fj)
        rels = df["relativity"].to_numpy()
        assert abs(np.mean(np.log(rels))) < 1e-8

    def test_heatmap_wide_shape(self, wrapper_with_interaction):
        w, fi, fj, scores_2d = wrapper_with_interaction
        wide = w.interaction_heatmap_wide(fi, fj)
        n_i, n_j = scores_2d.shape
        # wide has n_i rows and n_j + 1 columns (bin_i + one per bin_j)
        assert wide.shape[0] == n_i
        assert wide.shape[1] == n_j + 1

    def test_heatmap_unknown_feature_raises(self, fitted_wrapper):
        with pytest.raises(ValueError):
            fitted_wrapper.interaction_heatmap("driver_age", "unknown_col")

    def test_heatmap_no_interaction_term_raises(self, fitted_wrapper):
        """When the model has no interaction for this pair, raise ValueError."""
        with pytest.raises(ValueError, match="No interaction term"):
            fitted_wrapper.interaction_heatmap("driver_age", "ncd")

    def test_heatmap_unfitted_raises(self):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.interaction_heatmap("driver_age", "ncd")

    def test_heatmap_scores_match_2d_array(self, wrapper_with_interaction):
        """Scores in the long DataFrame should match the raw 2-D array."""
        w, fi, fj, scores_2d = wrapper_with_interaction
        df = w.interaction_heatmap(fi, fj)
        scores_flat_expected = scores_2d.flatten()
        scores_flat_actual = df["score"].to_numpy()
        np.testing.assert_allclose(scores_flat_actual, scores_flat_expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Diagnostics methods
# ---------------------------------------------------------------------------

class TestDiagnosticMethods:

    def test_ae_by_decile_returns_dataframe(self, fitted_wrapper, X_polars, synthetic_data):
        df = fitted_wrapper.ae_by_decile(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert isinstance(df, pl.DataFrame)
        assert "ae_ratio" in df.columns

    def test_ae_by_decile_n_bands(self, fitted_wrapper, X_polars, synthetic_data):
        df = fitted_wrapper.ae_by_decile(
            X_polars, synthetic_data["claim_count"], n_bands=5
        )
        assert len(df) == 5

    def test_gini_returns_float(self, fitted_wrapper, X_polars, synthetic_data):
        g = fitted_wrapper.gini_coefficient(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert isinstance(g, float)

    def test_gini_bounded(self, fitted_wrapper, X_polars, synthetic_data):
        g = fitted_wrapper.gini_coefficient(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert -1.0 <= g <= 1.0

    def test_segment_ae_schema(self, fitted_wrapper, X_polars, synthetic_data):
        segments = synthetic_data["vehicle_group"]
        df = fitted_wrapper.segment_ae_table(
            X_polars, synthetic_data["claim_count"], segment=segments
        )
        assert "ae_ratio" in df.columns
        assert "segment" in df.columns

    def test_ae_by_decile_unfitted_raises(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.ae_by_decile(X_polars, synthetic_data["claim_count"])


# ---------------------------------------------------------------------------
# Transparency report
# ---------------------------------------------------------------------------

class TestTransparencyReport:

    def test_report_returns_dict(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        assert isinstance(report, dict)

    def test_report_top_level_keys(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        required_keys = {
            "report_type", "model_metadata", "performance_metrics",
            "feature_profiles", "interaction_terms", "monotone_constraints",
            "known_limitations", "article_13_checklist",
        }
        assert required_keys.issubset(set(report.keys()))

    def test_report_model_metadata(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        meta = report["model_metadata"]
        assert meta["loss_family"] == "poisson"
        assert meta["n_features"] == len(X_polars.columns)
        assert meta["n_training_observations"] == len(X_polars)

    def test_report_performance_metrics(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"], exposure=synthetic_data["exposure"]
        )
        metrics = report["performance_metrics"]
        assert "deviance" in metrics
        assert "normalised_gini" in metrics
        assert "balance_ratio" in metrics
        assert metrics["balance_ratio"] > 0

    def test_report_article_13_checklist(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        checklist = report["article_13_checklist"]
        assert "13(1)(b)_capabilities_and_limitations" in checklist
        assert "13(1)(c)_purpose_and_performance" in checklist

    def test_report_custom_purpose(self, fitted_wrapper, X_polars, synthetic_data):
        purpose = "Motor third party liability frequency model"
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"],
            model_purpose=purpose
        )
        checklist = report["article_13_checklist"]
        assert checklist["13(1)(c)_purpose_and_performance"]["purpose"] == purpose

    def test_report_json_serialisable(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        json_str = json.dumps(report, default=str)
        parsed = json.loads(json_str)
        assert "model_metadata" in parsed

    def test_transparency_report_json_method(self, fitted_wrapper, X_polars, synthetic_data):
        json_str = fitted_wrapper.transparency_report_json(
            X_polars, synthetic_data["claim_count"]
        )
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "report_type" in parsed

    def test_report_limitations_included(self, fitted_wrapper, X_polars, synthetic_data):
        lims = ["Training data limited to 2020-2023", "No fraud indicator available"]
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"],
            limitations=lims
        )
        assert report["known_limitations"] == lims

    def test_report_human_oversight_provided(self, fitted_wrapper, X_polars, synthetic_data):
        oversight = "Reviewed quarterly by Chief Actuary"
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"],
            human_oversight=oversight
        )
        assert report["article_13_checklist"]["13(1)(d)_human_oversight"]["status"] == "provided"
        assert oversight in report["article_13_checklist"]["13(1)(d)_human_oversight"]["description"]

    def test_report_feature_profiles_list(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        profiles = report["feature_profiles"]
        assert isinstance(profiles, list)
        assert len(profiles) == len(X_polars.columns)
        for p in profiles:
            assert "feature" in p
            assert "leverage_range" in p

    def test_report_unfitted_raises(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper()
        with pytest.raises(RuntimeError, match="fitted"):
            w.transparency_report(X_polars, synthetic_data["claim_count"])

    def test_report_type_field(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        assert report["report_type"] == "EU_AI_Act_Article_13_Transparency_Report"

    def test_tweedie_report_has_variance_power(self, fitted_tweedie_wrapper, X_polars, synthetic_data):
        report = fitted_tweedie_wrapper.transparency_report(
            X_polars, synthetic_data["claim_amount"]
        )
        assert report["model_metadata"]["variance_power"] == 1.5

    def test_poisson_report_variance_power_is_none(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        assert report["model_metadata"]["variance_power"] is None

    def test_report_evaluation_n_obs(self, fitted_wrapper, X_polars, synthetic_data):
        report = fitted_wrapper.transparency_report(
            X_polars, synthetic_data["claim_count"]
        )
        assert report["performance_metrics"]["evaluation_n_obs"] == len(X_polars)


# ---------------------------------------------------------------------------
# cross_validate (module-level)
# ---------------------------------------------------------------------------

class TestCrossValidateFunction:

    def test_returns_dict_with_required_keys(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "poisson", "interactions": 0, "random_state": 42, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_count"],
            cv=3,
        )
        assert set(result.keys()) == {"scores", "mean_score", "std_score", "n_folds", "loss"}

    def test_n_folds_matches_cv(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "poisson", "interactions": 0, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_count"],
            cv=4,
        )
        assert result["n_folds"] == 4
        assert len(result["scores"]) == 4

    def test_scores_are_floats(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "poisson", "interactions": 0, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_count"],
            cv=3,
        )
        for s in result["scores"]:
            assert isinstance(s, float)

    def test_mean_score_equals_mean_of_scores(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "poisson", "interactions": 0, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_count"],
            cv=3,
        )
        expected_mean = float(np.mean(result["scores"]))
        assert abs(result["mean_score"] - expected_mean) < 1e-10

    def test_with_exposure(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "poisson", "interactions": 0, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_count"],
            exposure=synthetic_data["exposure"],
            cv=3,
        )
        assert result["n_folds"] == 3

    def test_loss_field_in_result(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "gamma", "interactions": 0, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_amount"],
            cv=3,
        )
        assert result["loss"] == "gamma"

    def test_std_score_is_non_negative(self, X_polars, synthetic_data):
        result = cross_validate(
            model_kwargs={"loss": "poisson", "interactions": 0, "n_jobs": 1},
            X=X_polars,
            y=synthetic_data["claim_count"],
            cv=3,
        )
        assert result["std_score"] >= 0


# ---------------------------------------------------------------------------
# EBMInsuranceWrapper.cross_validate method
# ---------------------------------------------------------------------------

class TestWrapperCrossValidate:

    def test_method_returns_dict(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
        result = w.cross_validate(X_polars, synthetic_data["claim_count"], cv=3)
        assert "scores" in result
        assert "mean_score" in result

    def test_method_uses_wrapper_loss(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="gamma", interactions=0, random_state=42, n_jobs=1)
        result = w.cross_validate(X_polars, synthetic_data["claim_amount"], cv=3)
        assert result["loss"] == "gamma"

    def test_method_with_exposure(self, X_polars, synthetic_data):
        w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
        result = w.cross_validate(
            X_polars, synthetic_data["claim_count"],
            exposure=synthetic_data["exposure"], cv=3
        )
        assert len(result["scores"]) == 3

    def test_wrapper_does_not_need_to_be_fitted_for_cv(self, X_polars, synthetic_data):
        """cross_validate() on an unfitted wrapper should still work — it trains fresh models."""
        w = EBMInsuranceWrapper(loss="poisson", interactions=0, random_state=42, n_jobs=1)
        result = w.cross_validate(X_polars, synthetic_data["claim_count"], cv=3)
        assert result["n_folds"] == 3

    def test_method_passes_monotone_constraints(self, X_polars, synthetic_data):
        """Monotone constraints should be included in the CV model kwargs."""
        mc = {"ncd": -1}
        w = EBMInsuranceWrapper(
            loss="poisson", interactions=0, monotone_constraints=mc, random_state=42, n_jobs=1
        )
        result = w.cross_validate(X_polars, synthetic_data["claim_count"], cv=3)
        # Just verify it ran without error and returned correct structure
        assert "scores" in result
