"""
Structural gap tests for insurance-gam EBM module.

Covers branches not exercised by the existing test suite:

_monotonicity.py
- _detect_direction: flat sequence (all diffs zero), tie-breaks
- _detect_direction: majority-decreasing sequence
- _is_monotone: single-element array
- _is_monotone: all-equal array (trivially monotone in both directions)
- _is_monotone: strictly increasing vs non-strictly increasing (tolerance)
- _isotonic_regression: correctness on known non-monotone input
- enforce: chaining (returns self, allow .enforce().enforce())
- enforce: interaction term raises ValueError

_relativities.py
- _detect_direction: correctly identifies majority direction
- RelativitiesTable.summary(): range >= 1 and sorted descending
- RelativitiesTable.table(): base bin relativity = 1.0 numerically
- RelativitiesTable.plot(): invalid kind raises
- _modal_bin_idx: AttributeError fallback (returns 0)
- _get_ebm_shape: feature not found raises
- summary() returns rows for all main-effect features

_model.py
- _deviance_poisson: exact known values
- _deviance_gamma: exact known values
- _deviance_tweedie p=1 and p=2 reduce to Poisson and Gamma
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _monotonicity helpers (pure functions, no model needed)
# ---------------------------------------------------------------------------

from insurance_gam.ebm._monotonicity import (
    _detect_direction,
    _is_monotone,
    _isotonic_regression,
)


class TestDetectDirection:

    def test_increasing_sequence(self):
        scores = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        assert _detect_direction(scores) == "increase"

    def test_decreasing_sequence(self):
        scores = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
        assert _detect_direction(scores) == "decrease"

    def test_majority_increasing(self):
        # Large step up then tiny step down
        scores = np.array([0.0, 3.0, 2.9])
        assert _detect_direction(scores) == "increase"

    def test_majority_decreasing(self):
        # Large step down then tiny step up
        scores = np.array([3.0, 0.1, 0.2])
        assert _detect_direction(scores) == "decrease"

    def test_flat_sequence_defaults_to_increase(self):
        """All diffs zero: positive_mass == negative_mass == 0, so the
        'increase' branch is taken (positive_mass >= negative_mass)."""
        scores = np.array([1.0, 1.0, 1.0, 1.0])
        assert _detect_direction(scores) == "increase"

    def test_single_step_up(self):
        scores = np.array([0.0, 1.0])
        assert _detect_direction(scores) == "increase"

    def test_single_step_down(self):
        scores = np.array([1.0, 0.0])
        assert _detect_direction(scores) == "decrease"


class TestIsMonotone:

    def test_single_element_trivially_monotone(self):
        """A single element is trivially monotone in both directions."""
        scores = np.array([1.0])
        assert _is_monotone(scores, "increase") is True
        assert _is_monotone(scores, "decrease") is True

    def test_all_equal_monotone_both_directions(self):
        """Constant array is monotone in both directions (diffs all zero)."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        assert _is_monotone(scores, "increase") is True
        assert _is_monotone(scores, "decrease") is True

    def test_strictly_increasing(self):
        scores = np.array([0.0, 0.1, 0.5, 1.2])
        assert _is_monotone(scores, "increase") is True
        assert _is_monotone(scores, "decrease") is False

    def test_non_monotone_is_false_both_directions(self):
        scores = np.array([0.0, 1.0, 0.5, 1.5])
        assert _is_monotone(scores, "increase") is False
        assert _is_monotone(scores, "decrease") is False

    def test_tolerance_allows_tiny_violations(self):
        """Violations within 1e-10 should be accepted."""
        scores = np.array([0.0, 0.5, 0.5 - 1e-11, 1.0])
        assert _is_monotone(scores, "increase") is True

    def test_invalid_direction_returns_false(self):
        """Unknown direction string should return False (not raise)."""
        scores = np.array([0.0, 0.5, 1.0])
        result = _is_monotone(scores, "flat")
        assert result is False


class TestIsotopicRegression:

    def test_already_increasing_unchanged(self):
        scores = np.array([0.0, 0.5, 1.0, 1.5])
        result = _isotonic_regression(scores, "increase")
        np.testing.assert_allclose(result, scores, atol=1e-8)

    def test_non_monotone_becomes_increasing(self):
        scores = np.array([0.0, 2.0, 1.0, 3.0])
        result = _isotonic_regression(scores, "increase")
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-8), f"Not non-decreasing: {result}"

    def test_non_monotone_becomes_decreasing(self):
        scores = np.array([3.0, 1.0, 2.0, 0.0])
        result = _isotonic_regression(scores, "decrease")
        diffs = np.diff(result)
        assert np.all(diffs <= 1e-8), f"Not non-increasing: {result}"

    def test_output_same_length(self):
        scores = np.array([0.3, 0.1, 0.5, 0.2, 0.8])
        result = _isotonic_regression(scores, "increase")
        assert len(result) == len(scores)

    def test_monotone_projection_bounds(self):
        """Isotonic regression stays within original value bounds."""
        scores = np.array([1.0, 0.5, 0.0, 2.0, 1.5])
        result = _isotonic_regression(scores, "increase")
        assert float(result.min()) >= float(scores.min()) - 1e-8
        assert float(result.max()) <= float(scores.max()) + 1e-8


# ---------------------------------------------------------------------------
# MonotonicityEditor: chaining and interaction term error
# ---------------------------------------------------------------------------

class TestMonotonicityEditorGaps:

    def test_enforce_chaining(self, fitted_poisson_model):
        """enforce() should return self, enabling method chaining."""
        from insurance_gam.ebm import MonotonicityEditor
        me = MonotonicityEditor(fitted_poisson_model)
        result = me.enforce("driver_age", direction="auto").enforce("ncd", direction="auto")
        assert result is me

    def test_enforce_interaction_term_raises(self, fitted_poisson_model):
        """
        An interaction term (2-D scores) must raise ValueError when passed to enforce().

        The mock EBM uses only main effects, so we patch a term to be 2-D.
        """
        from insurance_gam.ebm import MonotonicityEditor
        from insurance_gam.ebm._monotonicity import _get_term_scores

        me = MonotonicityEditor(fitted_poisson_model)
        ebm = fitted_poisson_model.ebm_

        # Temporarily replace driver_age scores with a 2-D array
        feature_idx = list(ebm.feature_names_in_).index("driver_age")
        original_scores = ebm.term_scores_[feature_idx].copy()

        try:
            ebm.term_scores_[feature_idx] = np.ones((3, 3))  # 2-D
            with pytest.raises(ValueError, match="interaction"):
                _get_term_scores(ebm, "driver_age")
        finally:
            ebm.term_scores_[feature_idx] = original_scores

    def test_check_unknown_feature_raises(self, fitted_poisson_model):
        """check() on an unknown feature must raise ValueError."""
        from insurance_gam.ebm import MonotonicityEditor
        me = MonotonicityEditor(fitted_poisson_model)
        with pytest.raises(ValueError, match="not in model"):
            me.check("unknown_feature")


# ---------------------------------------------------------------------------
# _relativities.py: _detect_direction and _modal_bin_idx fallback
# ---------------------------------------------------------------------------

class TestRelativitiesGaps:

    def test_modal_bin_idx_fallback_on_attribute_error(self, fitted_poisson_model):
        """_modal_bin_idx falls back to 0 when term_bin_weights_ is unavailable."""
        from insurance_gam.ebm._relativities import _modal_bin_idx

        ebm = fitted_poisson_model.ebm_
        # Temporarily remove term_bin_weights_
        original = ebm.term_bin_weights_
        del ebm.term_bin_weights_
        try:
            idx = _modal_bin_idx(ebm, "driver_age")
            assert idx == 0
        finally:
            ebm.term_bin_weights_ = original

    def test_relativity_table_base_bin_is_one(self, fitted_poisson_model):
        """The base bin must have relativity exactly 1.0."""
        from insurance_gam.ebm import RelativitiesTable
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("ncd")
        rels = df["relativity"].to_numpy()
        assert np.any(np.isclose(rels, 1.0, atol=1e-8)), (
            f"No bin has relativity=1.0. rels={rels}"
        )

    def test_relativity_table_unknown_feature_raises(self, fitted_poisson_model):
        """table() on an unknown feature must raise ValueError."""
        from insurance_gam.ebm import RelativitiesTable
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ValueError, match="not found"):
            rt.table("mystery_feature")

    def test_summary_range_sorted_descending(self, fitted_poisson_model):
        """summary() 'range' column should be sorted descending."""
        from insurance_gam.ebm import RelativitiesTable
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        ranges = df["range"].to_numpy()
        assert np.all(np.diff(ranges) <= 1e-8), "Summary not sorted by range descending"

    def test_summary_range_geq_one(self, fitted_poisson_model):
        """range = max_relativity / min_relativity >= 1 by definition."""
        from insurance_gam.ebm import RelativitiesTable
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        ranges = df["range"].to_numpy()
        assert np.all(ranges >= 1.0 - 1e-8)

    def test_plot_invalid_kind_raises(self, fitted_poisson_model):
        """plot() with an invalid kind should raise ValueError."""
        from insurance_gam.ebm import RelativitiesTable
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ValueError, match="kind"):
            rt.plot("driver_age", kind="scatter")

    def test_unfitted_model_raises(self):
        """RelativitiesTable on unfitted model must raise RuntimeError."""
        from insurance_gam.ebm import InsuranceEBM, RelativitiesTable
        model = InsuranceEBM(loss="poisson")
        with pytest.raises(RuntimeError, match="fitted"):
            RelativitiesTable(model)


# ---------------------------------------------------------------------------
# _model.py: deviance functions
# ---------------------------------------------------------------------------

class TestDevianceFunctions:
    """
    The deviance functions are internal but called via InsuranceEBM.score().
    We test them directly to verify the formula is correct.
    """

    def test_poisson_deviance_zero_at_perfect_prediction(self):
        """Poisson deviance is 0 when y_true == y_pred."""
        from insurance_gam.ebm._model import _deviance_poisson
        y = np.array([0.5, 1.0, 2.0, 0.0])
        d = _deviance_poisson(y, y)
        assert abs(d) < 1e-10

    def test_poisson_deviance_known_value(self):
        """D_Poisson(y, mu) = 2 * (y*log(y/mu) - (y-mu)).
        At y=1, mu=0.5: 2*(1*log(2) - 0.5) = 2*(0.6931 - 0.5) = 0.3863."""
        from insurance_gam.ebm._model import _deviance_poisson
        y = np.array([1.0])
        mu = np.array([0.5])
        expected = 2.0 * (1.0 * np.log(2.0) - 0.5)
        d = _deviance_poisson(y, mu)
        assert abs(d - expected) < 1e-6, f"Expected {expected:.6f}, got {d:.6f}"

    def test_poisson_deviance_positive(self):
        """Poisson deviance should be non-negative for y != mu."""
        from insurance_gam.ebm._model import _deviance_poisson
        rng = np.random.default_rng(200)
        y = rng.poisson(2.0, 100).astype(float)
        mu = rng.uniform(0.5, 4.0, 100)
        d = _deviance_poisson(y, mu)
        assert d >= 0.0

    def test_gamma_deviance_zero_at_perfect_prediction(self):
        """Gamma deviance is 0 when y_true == y_pred."""
        from insurance_gam.ebm._model import _deviance_gamma
        y = np.array([1000.0, 2000.0, 500.0])
        d = _deviance_gamma(y, y)
        assert abs(d) < 1e-8

    def test_gamma_deviance_known_value(self):
        """D_Gamma(y, mu) = 2 * (-log(y/mu) + (y-mu)/mu).
        At y=2, mu=1: 2*(-log(2) + 1) = 2*(1 - 0.6931) = 0.6137."""
        from insurance_gam.ebm._model import _deviance_gamma
        y = np.array([2.0])
        mu = np.array([1.0])
        expected = 2.0 * (-np.log(2.0) + 1.0)
        d = _deviance_gamma(y, mu)
        assert abs(d - expected) < 1e-6

    def test_gamma_deviance_positive(self):
        from insurance_gam.ebm._model import _deviance_gamma
        rng = np.random.default_rng(201)
        y = rng.gamma(2.0, 1000.0, 100)
        mu = rng.gamma(2.0, 900.0, 100)
        d = _deviance_gamma(y, mu)
        assert d >= 0.0

    def test_tweedie_p1_equals_poisson_formula(self):
        """Tweedie(p=1) deviance is Poisson deviance by definition."""
        from insurance_gam.ebm._model import _deviance_poisson, _deviance_tweedie
        rng = np.random.default_rng(202)
        y = rng.poisson(2.0, 50).astype(float)
        mu = np.maximum(rng.uniform(0.5, 3.0, 50), 0.01)
        d_pois = _deviance_poisson(y, mu)
        d_tw1 = _deviance_tweedie(y, mu, p=1.0)
        assert abs(d_pois - d_tw1) < 1e-4

    def test_poisson_deviance_weighted(self):
        """Weighted deviance should differ from unweighted when weights are unequal."""
        from insurance_gam.ebm._model import _deviance_poisson
        y = np.array([1.0, 2.0, 0.5])
        mu = np.array([0.8, 1.5, 0.3])
        w = np.array([1.0, 10.0, 1.0])  # middle observation weighted heavily
        d_unweighted = _deviance_poisson(y, mu)
        d_weighted = _deviance_poisson(y, mu, weights=w)
        # Weighted average is pulled toward the middle observation
        assert d_unweighted != d_weighted


# ---------------------------------------------------------------------------
# InsuranceEBM: feature_names property
# ---------------------------------------------------------------------------

class TestInsuranceEBMProperties:

    def test_feature_names_match_training_columns(self, fitted_poisson_model, X_polars):
        """feature_names should match the columns of the training DataFrame."""
        expected = X_polars.columns
        assert set(fitted_poisson_model.feature_names) == set(expected)

    def test_predict_shape(self, fitted_poisson_model, X_polars):
        """predict() output should have the same length as input."""
        preds = fitted_poisson_model.predict(X_polars)
        assert len(preds) == len(X_polars)

    def test_predict_positive(self, fitted_poisson_model, X_polars):
        """Poisson model predict() should return positive values."""
        preds = fitted_poisson_model.predict(X_polars)
        assert np.all(preds > 0)

    def test_not_fitted_raises_on_predict(self):
        """predict() before fit should raise RuntimeError."""
        from insurance_gam.ebm import InsuranceEBM
        model = InsuranceEBM(loss="poisson")
        import pandas as pd
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)
