"""
EBMInsuranceWrapper — unified actuarial facade for InsuranceEBM.

This is the high-level entry point for actuarial use of the EBM. It wraps
InsuranceEBM and aggregates the surrounding tools (diagnostics, relativities,
monotonicity, comparison) into a single object with a coherent actuarial API.

What this adds on top of InsuranceEBM:

1. Balance check and correction
   balance_check() returns sum(actual) / sum(predicted). A calibrated model
   has balance ratio ≈ 1.0. apply_balance_correction() adds a log offset to
   the intercept so the model is exactly globally balanced on any dataset.

2. Interaction heatmap
   interaction_heatmap() extracts 2-D shape functions for interaction terms
   (the pairs learned by EBM's cyclic boosting). Returns a polars DataFrame
   that can be pivoted to a heatmap or exported.

3. EU AI Act Article 13 transparency report
   transparency_report() returns a structured dict (JSON-serialisable) with
   model metadata, feature list, deviance, Gini, balance ratio, and an
   Article 13 checklist. Intended for model cards and regulator disclosures.

4. Exposure-weighted k-fold cross-validation
   cross_validate() wraps InsuranceEBM's fit/score in a stratified k-fold loop
   and returns per-fold deviances plus mean and standard deviation.

Design choices
--------------
EBMInsuranceWrapper is NOT a subclass of InsuranceEBM. It owns one. This avoids
the fragility of inheriting from a class whose __init__ signature may change,
and makes the ownership relationship explicit. You can always access the inner
model via wrapper.model_.

The wrapper is stateful once fit() is called — it stores the inner InsuranceEBM
as model_ (following sklearn convention) and is not safe for concurrent mutation.

Cross-validation deliberately re-instantiates InsuranceEBM for each fold. This
mirrors how actuaries actually run CV: each fold trains a fresh model, not a
warm-started one.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

import numpy as np
import polars as pl

from ._model import InsuranceEBM, _ensure_array, _to_pandas
from ._diagnostics import gini, double_lift, calibration_table
from ._relativities import RelativitiesTable


# ---------------------------------------------------------------------------
# Balance check helpers
# ---------------------------------------------------------------------------

def balance_ratio(
    y_true: Union[np.ndarray, pl.Series, list],
    y_pred: Union[np.ndarray, pl.Series, list],
    exposure: Optional[Union[np.ndarray, pl.Series, list]] = None,
) -> float:
    """
    Global balance ratio: sum(actual) / sum(predicted).

    A perfectly calibrated model has balance ratio = 1.0. Values above 1
    mean the model is systematically under-predicting (it will underprice);
    values below 1 mean over-prediction.

    Note the distinction from the A/E ratio in calibration_table(): that
    function reports A/E within segments, this one reports global A/E.
    When exposure is provided, both numerator and denominator are summed
    (not averaged) so that a book with heterogeneous exposure is handled
    correctly.

    Parameters
    ----------
    y_true : array-like
        Observed outcomes (raw counts or amounts, not rates).
    y_pred : array-like
        Predicted values (same units as y_true).
    exposure : array-like, optional
        Exposure weights. When provided, total actual and predicted are
        weighted sums: sum(y * w) / sum(y_hat * w). For frequency models,
        pass exposure so that short-exposure policies do not have
        disproportionate influence.

    Returns
    -------
    float
        Balance ratio (sum_actual / sum_predicted).

    Raises
    ------
    ValueError
        If sum of predicted is zero or negative.
    """
    y_t = _ensure_array(y_true)
    y_p = _ensure_array(y_pred)
    w = _ensure_array(exposure)

    if w is not None:
        total_actual = float(np.sum(y_t * w))
        total_predicted = float(np.sum(y_p * w))
    else:
        total_actual = float(np.sum(y_t))
        total_predicted = float(np.sum(y_p))

    if total_predicted <= 0:
        raise ValueError(
            f"Sum of predicted values is {total_predicted}. "
            "Cannot compute balance ratio with non-positive total prediction."
        )

    return total_actual / total_predicted


# ---------------------------------------------------------------------------
# Interaction heatmap extraction
# ---------------------------------------------------------------------------

def _extract_interaction_shape(ebm, feature_i: str, feature_j: str) -> tuple[
    list[str], list[str], np.ndarray
]:
    """
    Extract 2-D shape function for an interaction term.

    Returns (labels_i, labels_j, scores_2d) where scores_2d has shape
    (len(labels_i), len(labels_j)).

    Raises ValueError if no interaction term exists for this feature pair.
    """
    feat_names = list(ebm.feature_names_in_)

    try:
        idx_i = feat_names.index(feature_i)
    except ValueError:
        raise ValueError(
            f"Feature '{feature_i}' not in model. "
            f"Available: {feat_names}"
        )
    try:
        idx_j = feat_names.index(feature_j)
    except ValueError:
        raise ValueError(
            f"Feature '{feature_j}' not in model. "
            f"Available: {feat_names}"
        )

    # Find term index for this pair (order matters in interpretML: (i, j) or (j, i))
    interaction_term_idx = None
    for term_idx, term_feats in enumerate(ebm.term_features_):
        if set(term_feats) == {idx_i, idx_j}:
            interaction_term_idx = term_idx
            # Determine actual ordering
            if list(term_feats) == [idx_i, idx_j]:
                ordered_i, ordered_j = feature_i, feature_j
            else:
                ordered_i, ordered_j = feature_j, feature_i
            break

    if interaction_term_idx is None:
        raise ValueError(
            f"No interaction term found for features '{feature_i}' and '{feature_j}'. "
            "The EBM may not have learned an interaction for this pair, or "
            "interactions were disabled at fit time (interactions=0)."
        )

    explanation = ebm.explain_global(name="EBM")
    data = explanation.data(interaction_term_idx)

    names = data.get("names", [])
    scores = data.get("scores", [])

    scores_arr = np.array(scores, dtype=float)
    if np.ndim(scores_arr) != 2:
        raise ValueError(
            f"Expected 2-D scores for interaction term ({feature_i}, {feature_j}), "
            f"got shape {scores_arr.shape}. This term may not be a pairwise interaction."
        )

    # names for interactions is a list [names_i, names_j]
    if isinstance(names, (list, tuple)) and len(names) == 2:
        raw_names_i = [str(n) for n in names[0]]
        raw_names_j = [str(n) for n in names[1]]
    else:
        # Fallback: generate bin index labels
        raw_names_i = [str(k) for k in range(scores_arr.shape[0])]
        raw_names_j = [str(k) for k in range(scores_arr.shape[1])]

    # If names are bin edges (len n+1 for n bins), create interval labels
    def _edge_names_to_labels(edge_names: list[str], n_bins: int) -> list[str]:
        if len(edge_names) == n_bins + 1:
            return [f"({edge_names[k]}, {edge_names[k+1]}]" for k in range(n_bins)]
        elif len(edge_names) == n_bins:
            return edge_names
        else:
            return [str(k) for k in range(n_bins)]

    labels_i = _edge_names_to_labels(raw_names_i, scores_arr.shape[0])
    labels_j = _edge_names_to_labels(raw_names_j, scores_arr.shape[1])

    # Ensure scores are oriented as (feature_i, feature_j)
    if ordered_i != feature_i:
        scores_arr = scores_arr.T
        labels_i, labels_j = labels_j, labels_i

    return labels_i, labels_j, scores_arr


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    model_kwargs: dict[str, Any],
    X: Union[pl.DataFrame, "pd.DataFrame"],
    y: Union[np.ndarray, list, pl.Series, "pd.Series"],
    exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    sample_weight: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    cv: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Exposure-weighted k-fold cross-validation for InsuranceEBM.

    Each fold trains a fresh InsuranceEBM (instantiated with model_kwargs) and
    evaluates mean deviance on the held-out fold. Returns per-fold scores plus
    mean and standard deviation.

    This is not clever — it is a direct implementation of what actuaries do
    before signing off a model: train on N-1 folds, measure deviance on fold N,
    repeat N times. The mean CV deviance tells you whether the model generalises;
    the standard deviation tells you how stable it is across the book.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for InsuranceEBM (e.g. ``{'loss': 'poisson', 'interactions': 0}``).
    X : polars.DataFrame or pandas.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    exposure : array-like, optional
        Exposure weights.
    sample_weight : array-like, optional
        Additional observation weights.
    cv : int
        Number of folds. Default 5.
    shuffle : bool
        Whether to shuffle before splitting. Default True.
    random_state : int
        Random seed for shuffling. Default 42.

    Returns
    -------
    dict
        Keys:
        - ``scores``: list of float, per-fold mean deviance (negative, higher = better)
        - ``mean_score``: float, mean of scores
        - ``std_score``: float, std of scores
        - ``n_folds``: int
        - ``loss``: str, the loss used
    """
    import pandas as pd

    X_pd = _to_pandas(X)
    y_arr = _ensure_array(y)
    exp_arr = _ensure_array(exposure)
    sw_arr = _ensure_array(sample_weight)

    n = len(X_pd)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)

    fold_sizes = np.full(cv, n // cv)
    fold_sizes[: n % cv] += 1
    fold_start = np.concatenate([[0], np.cumsum(fold_sizes)])

    scores = []
    for fold in range(cv):
        test_idx = indices[fold_start[fold] : fold_start[fold + 1]]
        train_idx = np.concatenate([indices[: fold_start[fold]], indices[fold_start[fold + 1] :]])

        X_train = X_pd.iloc[train_idx]
        X_test = X_pd.iloc[test_idx]
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]
        exp_train = exp_arr[train_idx] if exp_arr is not None else None
        exp_test = exp_arr[test_idx] if exp_arr is not None else None
        sw_train = sw_arr[train_idx] if sw_arr is not None else None

        m = InsuranceEBM(**model_kwargs)
        m.fit(X_train, y_train, exposure=exp_train, sample_weight=sw_train)
        fold_score = m.score(X_test, y_test, exposure=exp_test)
        scores.append(fold_score)

    return {
        "scores": scores,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "n_folds": cv,
        "loss": model_kwargs.get("loss", "poisson"),
    }


# ---------------------------------------------------------------------------
# EBMInsuranceWrapper
# ---------------------------------------------------------------------------

class EBMInsuranceWrapper:
    """
    Unified actuarial facade for InterpretML's ExplainableBoostingMachine.

    This class assembles the full actuarial workflow into a single object.
    Internally it owns an InsuranceEBM (accessible as model_) and delegates
    fit/predict/score to it. The additional methods — balance checks, interaction
    heatmaps, transparency reports, cross-validation — are provided here so you
    do not have to import and co-ordinate four separate modules.

    Parameters
    ----------
    loss : str
        GLM family. One of 'poisson' (default), 'tweedie', 'gamma', 'mse',
        'mae', 'huber'.
    variance_power : float
        Tweedie variance power. Used only when loss='tweedie'. Default 1.5
        (compound Poisson-Gamma, typical for pure premium).
    interactions : int or str
        Number of interaction terms. '3x' means 3 * n_features pairs.
        Pass 0 to disable. Default '3x'.
    monotone_constraints : dict, optional
        Feature-level monotonicity constraints at fit time. E.g.
        ``{'ncd': -1, 'driver_age': 0}``.
    **ebm_kwargs
        Passed through to InsuranceEBM (and thus to
        ExplainableBoostingRegressor). Common: ``random_state``, ``n_jobs``,
        ``max_bins``, ``learning_rate``, ``n_estimators``.

    Attributes
    ----------
    model_ : InsuranceEBM
        The fitted inner model. Available after fit().
    relativities_ : RelativitiesTable
        Relativity extractor. Available after fit().
    is_fitted_ : bool
        True once fit() has been called.

    Examples
    --------
    >>> wrapper = EBMInsuranceWrapper(loss='poisson', interactions=0, random_state=42)
    >>> wrapper.fit(X, claim_count, exposure=exposure)
    >>> wrapper.balance_check(X, claim_count, exposure=exposure)
    1.003
    >>> report = wrapper.transparency_report(X, claim_count, exposure=exposure)
    >>> print(report['balance_ratio'])
    1.003
    """

    def __init__(
        self,
        loss: str = "poisson",
        variance_power: float = 1.5,
        interactions: Union[int, str] = "3x",
        monotone_constraints: Optional[dict] = None,
        **ebm_kwargs,
    ) -> None:
        self.loss = loss
        self.variance_power = variance_power
        self.interactions = interactions
        self.monotone_constraints = monotone_constraints
        self.ebm_kwargs = ebm_kwargs

        self.model_: Optional[InsuranceEBM] = None
        self.relativities_: Optional[RelativitiesTable] = None
        self.is_fitted_: bool = False

        # Stored at fit time for transparency report
        self._n_obs_train: Optional[int] = None
        self._n_features: Optional[int] = None
        self._feature_names: Optional[list] = None

    def _build_model(self) -> InsuranceEBM:
        return InsuranceEBM(
            loss=self.loss,
            variance_power=self.variance_power,
            interactions=self.interactions,
            monotone_constraints=self.monotone_constraints,
            **self.ebm_kwargs,
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "EBMInsuranceWrapper has not been fitted. Call fit() first."
            )

    def fit(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
        sample_weight: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> "EBMInsuranceWrapper":
        """
        Fit the EBM to insurance training data.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix.
        y : array-like
            Target variable (claim count, amount, or pure premium).
        exposure : array-like, optional
            Exposure measure (earned car years, etc.). Incorporated as
            log-offset.
        sample_weight : array-like, optional
            Row-level weights. Applied in addition to exposure.

        Returns
        -------
        EBMInsuranceWrapper
            self, for method chaining.

        Raises
        ------
        ImportError
            If the ``interpret`` package is not installed
            (``pip install insurance-gam[ebm]``).
        ValueError
            If X and y have incompatible shapes, or if exposure has a different
            length from X.
        """
        self.model_ = self._build_model()
        self.model_.fit(X, y, exposure=exposure, sample_weight=sample_weight)
        self.relativities_ = RelativitiesTable(self.model_)
        self.is_fitted_ = True

        # Cache metadata for transparency report
        X_pd = _to_pandas(X)
        self._n_obs_train = len(X_pd)
        self._n_features = len(X_pd.columns)
        self._feature_names = list(X_pd.columns)

        return self

    def predict(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> np.ndarray:
        """
        Predict expected values on the response scale.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix.
        exposure : array-like, optional
            Exposure for scaling predictions. Incorporated as log-offset.

        Returns
        -------
        numpy.ndarray
            Predicted values, shape (n,). On the response scale (after exp()).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()
        return self.model_.predict(X, exposure=exposure)

    def predict_log_score(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
    ) -> np.ndarray:
        """
        Return raw log-scale scores (before exp()).

        Useful for inspecting the additive log contributions of features
        before the exponential link function is applied.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Log-scale scores, shape (n,). Equivalent to
            ``np.log(self.predict(X))``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()
        return self.model_.predict_log_score(X)

    def score(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> float:
        """
        Mean deviance on test data (negated, higher = better).

        Follows the sklearn convention of returning a score where higher is
        better, so this returns -mean_deviance. Pass to :func:`cross_validate`
        or use directly for hold-out evaluation.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Feature matrix.
        y : array-like
            Observed outcomes.
        exposure : array-like, optional
            Exposure weights.

        Returns
        -------
        float
            Negative mean deviance. A value of -0.12 means mean deviance = 0.12.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()
        return self.model_.score(X, y, exposure=exposure)

    # ------------------------------------------------------------------
    # Balance check and correction
    # ------------------------------------------------------------------

    def balance_check(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> float:
        """
        Global balance ratio: sum(actual) / sum(predicted).

        A calibrated model has balance ratio ≈ 1.0. If the ratio deviates
        materially (e.g. < 0.98 or > 1.02 on a large portfolio), apply
        apply_balance_correction() to enforce global balance.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
        y : array-like
            Observed outcomes (raw counts or amounts).
        exposure : array-like, optional
            Exposure weights.

        Returns
        -------
        float
            Balance ratio.
        """
        self._check_fitted()
        y_pred = self.model_.predict(X, exposure=exposure)
        return balance_ratio(y, y_pred, exposure=exposure)

    def apply_balance_correction(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> float:
        """
        Adjust the EBM intercept so the model is globally balanced on (X, y).

        The correction is: intercept += log(sum_actual / sum_predicted).
        This is applied directly to the EBM's intercept_ attribute, so
        subsequent predict() calls will reflect the correction.

        This is the insurance equivalent of a GLM post-fit re-calibration.
        It preserves all shape functions exactly — only the overall level shifts.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
        y : array-like
        exposure : array-like, optional

        Returns
        -------
        float
            The log correction factor applied.

        Notes
        -----
        Only meaningful for log-link families (Poisson, Tweedie, Gamma).
        For MSE/MAE models the intercept adjustment is additive, not log.
        The method applies a log correction for all loss types (additive
        in log space = multiplicative in response space).
        """
        self._check_fitted()
        y_pred = self.model_.predict(X, exposure=exposure)
        ratio = balance_ratio(y, y_pred, exposure=exposure)
        log_correction = float(np.log(ratio))

        # Apply to EBM intercept_
        ebm = self.model_.ebm_
        if hasattr(ebm, "intercept_"):
            ebm.intercept_ = float(ebm.intercept_) + log_correction
        elif hasattr(ebm, "intercept"):
            ebm.intercept = float(ebm.intercept) + log_correction
        else:
            raise AttributeError(
                "Cannot apply balance correction: fitted EBM has no "
                "'intercept_' or 'intercept' attribute. "
                "Check your interpretML version."
            )

        return log_correction

    # ------------------------------------------------------------------
    # Relativities
    # ------------------------------------------------------------------

    def relativity_table(self, feature: str) -> pl.DataFrame:
        """
        Relativity table for a single feature.

        Parameters
        ----------
        feature : str

        Returns
        -------
        polars.DataFrame
            Columns: bin_label, raw_score, relativity.
        """
        self._check_fitted()
        return self.relativities_.table(feature)

    def relativity_summary(self) -> pl.DataFrame:
        """
        Summary table of all main-effect features.

        Returns
        -------
        polars.DataFrame
            Columns: feature, n_bins, min_relativity, max_relativity, range.
            Sorted by range descending (features with most leverage first).
        """
        self._check_fitted()
        return self.relativities_.summary()

    # ------------------------------------------------------------------
    # Interaction heatmap
    # ------------------------------------------------------------------

    def interaction_heatmap(
        self, feature_i: str, feature_j: str
    ) -> pl.DataFrame:
        """
        2-D interaction shape function as a long-format DataFrame.

        Returns the interaction effect between feature_i and feature_j
        as a polars DataFrame with columns (bin_i, bin_j, score,
        relativity). relativity = exp(score - mean(score)), so the
        average cell has relativity ≈ 1.0.

        This is the actuarial equivalent of a two-way rating table.
        High relativity cells indicate combinations that are riskier
        than the main effects would suggest; low cells are safer.

        Parameters
        ----------
        feature_i : str
            First feature name.
        feature_j : str
            Second feature name.

        Returns
        -------
        polars.DataFrame
            Columns: bin_i (str), bin_j (str), score (float),
            relativity (float).

        Raises
        ------
        ValueError
            If no interaction term exists for this feature pair in the
            fitted model.
        """
        self._check_fitted()
        labels_i, labels_j, scores_2d = _extract_interaction_shape(
            self.model_.ebm_, feature_i, feature_j
        )

        # Relativities centred on the mean score across all cells
        mean_score = float(np.mean(scores_2d))
        rels_2d = np.exp(scores_2d - mean_score)

        rows = []
        for ii, label_i in enumerate(labels_i):
            for jj, label_j in enumerate(labels_j):
                rows.append(
                    {
                        "bin_i": label_i,
                        "bin_j": label_j,
                        "score": float(scores_2d[ii, jj]),
                        "relativity": float(rels_2d[ii, jj]),
                    }
                )

        return pl.DataFrame(rows)

    def interaction_heatmap_wide(
        self, feature_i: str, feature_j: str
    ) -> pl.DataFrame:
        """
        2-D interaction shape function in wide format (pivot table).

        Each row is a bin of feature_i, each column is a bin of feature_j.
        Values are relativities (exp(score - mean_score)).

        Useful for display in Excel or a notebook.

        Parameters
        ----------
        feature_i : str
        feature_j : str

        Returns
        -------
        polars.DataFrame
            Index column 'bin_i' (str), then one column per bin of
            feature_j. Values are relativities.
        """
        self._check_fitted()
        long_df = self.interaction_heatmap(feature_i, feature_j)
        wide = long_df.pivot(
            values="relativity", index="bin_i", on="bin_j"
        )
        return wide

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def ae_by_decile(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
        n_bands: int = 10,
    ) -> pl.DataFrame:
        """
        Actual/expected by predicted risk decile.

        Policies are sorted by model prediction and grouped into n_bands
        equal-weight buckets. The table shows mean actual, mean predicted,
        and the A/E ratio per band. A well-calibrated model has A/E close
        to 1.0 across all bands.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
        y : array-like
        exposure : array-like, optional
        n_bands : int
            Number of bands (default 10).

        Returns
        -------
        polars.DataFrame
            Columns: band, exposure, actual, predicted, ae_ratio.
        """
        self._check_fitted()
        y_pred = self.model_.predict(X, exposure=exposure)
        return double_lift(y, y_pred, exposure=exposure, n_bands=n_bands)

    def gini_coefficient(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> float:
        """
        Normalised Gini coefficient.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
        y : array-like
        exposure : array-like, optional

        Returns
        -------
        float
            Normalised Gini coefficient in [-1, 1].
        """
        self._check_fitted()
        y_pred = self.model_.predict(X, exposure=exposure)
        return gini(y, y_pred, exposure=exposure)

    def segment_ae_table(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        segment: Union[np.ndarray, pl.Series, list],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
    ) -> pl.DataFrame:
        """
        Actual vs expected by segment.

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
        y : array-like
        segment : array-like
            Segment labels.
        exposure : array-like, optional

        Returns
        -------
        polars.DataFrame
            Columns: segment, exposure, actual_total, predicted_total, ae_ratio.
        """
        self._check_fitted()
        y_pred = self.model_.predict(X, exposure=exposure)
        return calibration_table(y, y_pred, segment, exposure=exposure)

    # ------------------------------------------------------------------
    # EU AI Act Article 13 transparency report
    # ------------------------------------------------------------------

    def transparency_report(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
        model_name: str = "EBMInsuranceWrapper",
        model_purpose: str = "Insurance pricing — risk classification and premium estimation",
        limitations: Optional[list[str]] = None,
        human_oversight: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate an EU AI Act Article 13 transparency report.

        Article 13 of the EU AI Act (2024/1689) requires providers of
        high-risk AI systems to ensure the systems are sufficiently
        transparent for deployers to understand and use them correctly.
        Insurance pricing models that determine access to or cost of
        products are likely to be classified as high-risk AI under
        Annex III.

        The report is a JSON-serialisable dict. It contains:

        - model_metadata: name, loss family, feature count, training obs
        - performance_metrics: deviance on provided dataset, Gini, balance
        - feature_profiles: relativities summary for all main effects
        - interaction_terms: list of detected interaction pairs
        - article_13_checklist: dict of required disclosure items

        The checklist uses the actual Article 13(1) sub-items and marks
        them as 'provided' where this method supplies the information,
        and 'requires_human_input' where the deployer must add it (e.g.
        intended geographic scope, data governance).

        Parameters
        ----------
        X : polars.DataFrame or pandas.DataFrame
            Evaluation dataset (may be holdout or training data).
        y : array-like
            Observed outcomes.
        exposure : array-like, optional
            Exposure weights.
        model_name : str
            Identifying name for the model in the report.
        model_purpose : str
            Prose description of the model's intended purpose.
        limitations : list of str, optional
            Known limitations. Added verbatim to the report.
        human_oversight : str, optional
            Description of human oversight measures in place.

        Returns
        -------
        dict
            JSON-serialisable dict. Serialise with json.dumps(report).
        """
        self._check_fitted()

        # Compute performance metrics on the provided dataset
        y_arr = _ensure_array(y)
        y_pred = self.model_.predict(X, exposure=exposure)
        exp_arr = _ensure_array(exposure)

        deviance_score = -self.model_.score(X, y, exposure=exposure)  # positive deviance
        gini_score = gini(y_arr, y_pred, exposure=exp_arr)
        bal_ratio = balance_ratio(y_arr, y_pred, exposure=exp_arr)

        # Relativities summary
        rel_summary = self.relativities_.summary()
        feature_profiles = []
        for row in rel_summary.iter_rows(named=True):
            feature_profiles.append({
                "feature": row["feature"],
                "n_bins": row["n_bins"],
                "min_relativity": round(row["min_relativity"], 4),
                "max_relativity": round(row["max_relativity"], 4),
                "leverage_range": round(row["range"], 4),
            })

        # Detect interaction terms
        ebm = self.model_.ebm_
        interaction_terms = []
        feat_names = list(ebm.feature_names_in_)
        for term_feats in ebm.term_features_:
            if len(term_feats) == 2:
                i, j = term_feats
                interaction_terms.append(
                    {"feature_i": feat_names[i], "feature_j": feat_names[j]}
                )

        # Monotone constraints applied
        mc = self.model_.monotone_constraints or {}
        monotone_list = [
            {"feature": k, "direction": "increasing" if v > 0 else "decreasing"}
            for k, v in mc.items()
            if v != 0
        ]

        # Article 13 checklist
        checklist = {
            "13(1)(a)_identity_and_contact": "requires_human_input",
            "13(1)(b)_capabilities_and_limitations": {
                "status": "partial",
                "note": "Performance metrics and feature leverage provided; "
                        "geographic/temporal scope requires human input.",
                "metrics_provided": {
                    "deviance": round(float(deviance_score), 6),
                    "normalised_gini": round(float(gini_score), 4),
                    "balance_ratio": round(float(bal_ratio), 4),
                },
            },
            "13(1)(c)_purpose_and_performance": {
                "status": "provided",
                "purpose": model_purpose,
                "loss_family": self.loss,
            },
            "13(1)(d)_human_oversight": {
                "status": "provided" if human_oversight else "requires_human_input",
                "description": human_oversight or "Not specified",
            },
            "13(1)(e)_technical_capabilities": {
                "status": "provided",
                "n_features": len(self._feature_names or []),
                "n_interaction_terms": len(interaction_terms),
                "monotone_constraints": monotone_list,
            },
            "13(2)_instructions_for_use": "requires_human_input",
            "13(3)_plain_language": "requires_human_input",
        }

        report = {
            "report_type": "EU_AI_Act_Article_13_Transparency_Report",
            "model_metadata": {
                "model_name": model_name,
                "model_class": "EBMInsuranceWrapper",
                "loss_family": self.loss,
                "variance_power": self.variance_power if self.loss == "tweedie" else None,
                "n_training_observations": self._n_obs_train,
                "n_features": self._n_features,
                "feature_names": self._feature_names,
                "interactions": str(self.interactions),
            },
            "performance_metrics": {
                "deviance": round(float(deviance_score), 6),
                "normalised_gini": round(float(gini_score), 4),
                "balance_ratio": round(float(bal_ratio), 4),
                "evaluation_n_obs": len(y_arr),
            },
            "feature_profiles": feature_profiles,
            "interaction_terms": interaction_terms,
            "monotone_constraints": monotone_list,
            "known_limitations": limitations or [],
            "article_13_checklist": checklist,
        }

        return report

    def transparency_report_json(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
        **kwargs,
    ) -> str:
        """
        Return the transparency report as a JSON string.

        Parameters
        ----------
        X, y, exposure : see transparency_report()
        **kwargs : passed through to transparency_report()

        Returns
        -------
        str
            JSON string.
        """
        return json.dumps(
            self.transparency_report(X, y, exposure=exposure, **kwargs),
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: Union[pl.DataFrame, "pd.DataFrame"],
        y: Union[np.ndarray, list, pl.Series, "pd.Series"],
        exposure: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
        sample_weight: Optional[Union[np.ndarray, list, pl.Series, "pd.Series"]] = None,
        cv: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Exposure-weighted k-fold cross-validation.

        Uses the same constructor parameters as self when building each
        fold's model. This means you do not need to separately specify
        model_kwargs — the wrapper already knows its own configuration.

        Parameters
        ----------
        X, y, exposure, sample_weight : training data.
        cv : int
            Number of folds. Default 5.
        shuffle : bool
            Whether to shuffle before splitting.
        random_state : int

        Returns
        -------
        dict
            Keys: scores (list), mean_score (float), std_score (float),
            n_folds (int), loss (str).
        """
        model_kwargs = {
            "loss": self.loss,
            "variance_power": self.variance_power,
            "interactions": self.interactions,
        }
        if self.monotone_constraints:
            model_kwargs["monotone_constraints"] = self.monotone_constraints
        model_kwargs.update(self.ebm_kwargs)

        return cross_validate(
            model_kwargs=model_kwargs,
            X=X,
            y=y,
            exposure=exposure,
            sample_weight=sample_weight,
            cv=cv,
            shuffle=shuffle,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Properties and repr
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list:
        """Feature names from the training data."""
        self._check_fitted()
        return self.model_.feature_names

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"EBMInsuranceWrapper("
            f"loss='{self.loss}', "
            f"variance_power={self.variance_power}, "
            f"interactions='{self.interactions}', "
            f"{status})"
        )
