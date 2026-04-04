"""
BoulevardEBM and BoulevardInference — moving-average boosted stumps with
analytical CLT-based confidence intervals on shape functions.

Implements Algorithm 2 from:

    Fang et al. (2026). "Boulevard: Regularized Stochastic Gradient Boosted
    Trees and Their Limiting Distribution." AISTATS 2026. arXiv:2601.18857.

The key insight is that the moving-average update (f[k] = ((b-1)/b)*f[k] +
(lambda/b)*new_stump) causes the fitted shape functions to converge to a
kernel ridge regression solution as n_estimators → ∞. This gives us:
  - A known asymptotic distribution for each shape function estimate
  - Analytical confidence intervals that scale with a hat-matrix diagonal
    computed entirely in bin space — O(m²) not O(n²) where m ≤ 256

Only MSE/Gaussian loss is supported in v1. Poisson/Tweedie variants would
require a reweighting step at each boosting round; document as future work.

Design choices:
  - No dependency on interpretML. Stumps are sklearn DecisionTreeRegressor
    with max_depth=1.
  - Polars output DataFrames (consistent with rest of insurance-gam).
  - All kernel matrix operations in bin space, never in sample space.
    For m=256 bins and k features the kernel inversion is k independent
    256×256 solves — negligible compute.
  - Equal-frequency binning via numpy quantiles so that each bin has roughly
    equal weight. This makes the bin-frequency matrix approximately uniform,
    which is numerically better conditioned for the kernel inversion.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import linalg
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_2d(X: Union[np.ndarray, "pl.DataFrame"]) -> tuple[np.ndarray, list[str]]:
    """Return (array, feature_names) from numpy array or polars DataFrame."""
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(float), list(X.columns)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, [f"x{i}" for i in range(arr.shape[1])]


def _equal_freq_bins(values: np.ndarray, max_bins: int) -> np.ndarray:
    """
    Compute equal-frequency bin edges for a 1-D feature.

    Returns an array of length min(max_bins+1, n_unique+1) giving the
    left-closed, right-open bin boundaries.  The final edge is set to
    +inf so that the rightmost bin captures the maximum value.
    """
    n_bins = min(max_bins, len(np.unique(values)))
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(values, quantiles)
    # Deduplicate (happens when many ties exist)
    edges = np.unique(edges)
    # Make final edge open by replacing it with +inf
    edges[-1] = np.inf
    return edges


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Map values to 0-based bin indices using the provided edges.

    Bins are left-closed: bin j contains values in [edges[j], edges[j+1]).
    Returns integer indices in [0, len(edges)-2].
    """
    # np.searchsorted with 'right' gives the insertion point after duplicates,
    # matching left-closed semantics when we subtract 1.
    idx = np.searchsorted(edges[:-1], values, side="right") - 1
    # Clip to valid range in case of floating-point edge cases
    return np.clip(idx, 0, len(edges) - 2).astype(np.intp)


# ---------------------------------------------------------------------------
# BoulevardEBM
# ---------------------------------------------------------------------------

class BoulevardEBM:
    """
    Boosted-stump GAM with moving-average (Boulevard) regularisation.

    Fits an additive model  f(x) = intercept + sum_k f_k(x_k)  where each
    shape function f_k is a step function over equal-frequency bins.  The
    moving-average update ensures the estimator converges to a kernel ridge
    regression solution, enabling analytical CLT-based confidence intervals
    via BoulevardInference.

    Only MSE (Gaussian) loss is supported.  For count or severity modelling
    on an insurance book, fit on the log scale or use the Poisson-deviance
    EBM in InsuranceEBM.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (B in the paper).  More rounds → closer
        to the KRR limit and wider CIs (the MA average variance decays as
        1/B).  1000 is a reasonable default; increase to 5000 for smoother
        shape functions on small datasets.
    lambda_ : float
        Ridge regularisation strength (λ in Algorithm 2).  Larger values
        shrink shape functions toward zero and produce narrower, more biased
        CIs.  Default 1.0 is a sensible starting point; use cv_lambda() to
        select automatically.
    max_bins : int
        Maximum number of equal-frequency bins per feature.  256 matches
        the interpretML / LightGBM default and gives a good resolution vs.
        computation trade-off.
    min_samples_leaf : int
        Minimum samples per leaf in the depth-1 decision stump.  Acts as a
        secondary smoothing lever beyond lambda_.
    random_state : int
        Random seed for reproducibility.

    Attributes
    ----------
    feature_names_ : list[str]
        Feature names from the training data.
    bins_ : list[np.ndarray]
        Bin edge arrays, one per feature.  Length of each array is
        n_bins_k + 1 with the last edge set to +inf.
    shape_scores_ : list[np.ndarray]
        Fitted shape function values, one array per feature.  The j-th
        element is the predicted contribution for a sample in bin j.
    intercept_ : float
        Global intercept (mean of y after removing all shape contributions).
    sigma_hat_ : float
        Estimated residual standard deviation used by BoulevardInference for
        SE calculation.  Set from training residuals by fit(), but overridden
        with the OOS estimate when cv_lambda() is used (see sigma_hat_oos_).
    sigma_hat_oos_ : float or None
        Out-of-sample residual standard deviation from the CV folds run by
        cv_lambda().  None if cv_lambda() has not been called.  This estimate
        is less biased than the training-residual version because it does not
        include KRR approximation error.
    n_samples_ : int
        Number of training samples.
    bin_counts_ : list[np.ndarray]
        Number of training samples in each bin, one array per feature.
    cv_results_ : dict or None
        Cross-validation results from cv_lambda(), containing keys
        ``lambda_values`` and ``mean_cv_mse``.  None if cv_lambda() has not
        been called.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        lambda_: float = 1.0,
        max_bins: int = 256,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # Fitted attributes — set by fit()
        self.feature_names_: list[str] = []
        self.bins_: list[np.ndarray] = []
        self.shape_scores_: list[np.ndarray] = []
        self.intercept_: float = 0.0
        self.sigma_hat_: float = 1.0
        self.sigma_hat_oos_: Optional[float] = None
        self.n_samples_: int = 0
        self.bin_counts_: list[np.ndarray] = []
        self.cv_results_: Optional[dict] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],
        y: Union[np.ndarray, list],
    ) -> "BoulevardEBM":
        """
        Fit the Boulevard EBM to training data.

        Parameters
        ----------
        X : np.ndarray or polars.DataFrame
            Feature matrix, shape (n, p).  If a polars DataFrame, column
            names are used as feature names.
        y : array-like
            Target variable.  MSE loss — fit on log scale for multiplicative
            models (e.g. log(pure_premium)).

        Returns
        -------
        self
        """
        X_arr, feature_names = _to_numpy_2d(X)
        y_arr = np.asarray(y, dtype=float)
        n, p = X_arr.shape

        self.feature_names_ = feature_names
        self.n_samples_ = n

        # 1. Build bins for each feature
        self.bins_ = [_equal_freq_bins(X_arr[:, k], self.max_bins) for k in range(p)]

        # 2. Pre-compute bin indices for every sample (n × p integer matrix)
        bin_indices = np.column_stack(
            [_digitize(X_arr[:, k], self.bins_[k]) for k in range(p)]
        )  # shape (n, p)

        # 3. Bin counts — used by BoulevardInference for the kernel matrix
        self.bin_counts_ = []
        for k in range(p):
            n_bins_k = len(self.bins_[k]) - 1
            counts = np.bincount(bin_indices[:, k], minlength=n_bins_k)
            self.bin_counts_.append(counts.astype(float))

        # 4. Initialise shape scores at zero, intercept at y-mean
        self.intercept_ = float(np.mean(y_arr))
        f = [np.zeros(len(self.bins_[k]) - 1) for k in range(p)]

        # Working residuals relative to the current full model
        # r_i = y_i - intercept - sum_k f_k(x_{ik})
        # We maintain the vector of current contributions per feature:
        # contrib[k][i] = f[k][bin_indices[i, k]]
        contrib = [f[k][bin_indices[:, k]] for k in range(p)]

        # Total fitted value (excluding intercept, included separately below)
        fitted = sum(contrib)  # shape (n,)

        stump = DecisionTreeRegressor(
            max_depth=1,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        # 5. Boosting loop — Algorithm 2 from Fang et al.
        for b in range(1, self.n_estimators + 1):
            for k in range(p):
                # Residuals excluding feature k's contribution
                r_k = y_arr - self.intercept_ - fitted + contrib[k]  # shape (n,)

                # Fit depth-1 stump to residuals; use bin index as the sole
                # feature so that the stump's split is always at a bin boundary
                stump.fit(bin_indices[:, k : k + 1], r_k)
                t_raw = stump.predict(bin_indices[:, k : k + 1])

                # Aggregate predictions to bin-level scores (mean prediction
                # for each bin from the stump)
                n_bins_k = len(self.bins_[k]) - 1
                bin_sums = np.bincount(
                    bin_indices[:, k], weights=t_raw, minlength=n_bins_k
                )
                bin_cnts = np.bincount(
                    bin_indices[:, k], minlength=n_bins_k
                ).astype(float)
                with np.errstate(invalid="ignore"):
                    t_bin = np.where(bin_cnts > 0, bin_sums / bin_cnts, 0.0)

                # Centre the bin-level stump scores
                t_centred = t_bin - np.mean(t_bin)

                # Moving-average update:
                # f[k] = ((b-1)/b) * f[k] + (lambda/b) * t_centred
                f[k] = ((b - 1) / b) * f[k] + (self.lambda_ / b) * t_centred

                # Update contributions
                contrib[k] = f[k][bin_indices[:, k]]

            # Recompute total fitted value after the inner k-loop
            fitted = sum(contrib)

        self.shape_scores_ = f

        # 6. Estimate residual standard deviation on training data
        residuals = y_arr - self.intercept_ - fitted
        n_params = sum(len(fk) for fk in f)  # approximate degrees of freedom
        denom = max(n - n_params - 1, 1)
        self.sigma_hat_ = float(np.sqrt(np.sum(residuals ** 2) / denom))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Cross-validation for lambda selection
    # ------------------------------------------------------------------

    @classmethod
    def cv_lambda(
        cls,
        X: Union[np.ndarray, "pl.DataFrame"],
        y: Union[np.ndarray, list],
        lambda_grid: Optional[np.ndarray] = None,
        n_folds: int = 5,
        n_estimators: int = 1000,
        max_bins: int = 256,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> "BoulevardEBM":
        """
        Select the regularisation parameter lambda via k-fold cross-validation,
        then refit on the full dataset.

        For each candidate lambda, the model is fitted on (k-1) folds and
        evaluated on the held-out fold using MSE.  The lambda with the lowest
        mean CV MSE is selected.  The final model is then fitted to the full
        dataset using that lambda.

        Out-of-sample (OOS) residuals from the CV folds are used to compute
        sigma_hat_ on the returned model.  This estimate is less biased than
        the training-residual version because it does not include KRR
        approximation error — see sigma_hat_oos_ vs sigma_hat_ for both values.

        Parameters
        ----------
        X : np.ndarray or polars.DataFrame
            Feature matrix, shape (n, p).
        y : array-like
            Target variable.
        lambda_grid : np.ndarray, optional
            1-D array of candidate lambda values.  Defaults to
            ``np.logspace(-3, 3, 20)`` (20 values from 0.001 to 1000).
        n_folds : int
            Number of CV folds.  Default 5.
        n_estimators : int
            Boosting rounds for each CV fit.  Default 1000.
        max_bins : int
            Maximum bins per feature.  Default 256.
        min_samples_leaf : int
            Minimum leaf size for each stump.  Default 20.
        random_state : int
            Random seed for KFold splitting and stump fitting.

        Returns
        -------
        BoulevardEBM
            A new BoulevardEBM instance fitted on the full dataset with the
            best lambda.  Two additional attributes are set:
            - ``cv_results_`` : dict with keys ``lambda_values`` (list of
              floats) and ``mean_cv_mse`` (list of floats).
            - ``sigma_hat_oos_`` : float — OOS residual std from CV folds.
              ``sigma_hat_`` is also overridden with this value so that
              BoulevardInference uses the less-biased estimate automatically.

        Examples
        --------
        >>> model = BoulevardEBM.cv_lambda(X_train, y_train)
        >>> print(model.lambda_)   # best lambda chosen by CV
        >>> print(model.cv_results_)
        """
        if lambda_grid is None:
            lambda_grid = np.logspace(-3, 3, 20)

        X_arr, _ = _to_numpy_2d(X)
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(kf.split(X_arr))

        mean_cv_mse: list[float] = []

        for lam in lambda_grid:
            fold_mses: list[float] = []
            for train_idx, val_idx in folds:
                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

                m = cls(
                    n_estimators=n_estimators,
                    lambda_=float(lam),
                    max_bins=max_bins,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                )
                m.fit(X_tr, y_tr)
                preds_va = m.predict(X_va)
                fold_mses.append(float(np.mean((y_va - preds_va) ** 2)))

            mean_cv_mse.append(float(np.mean(fold_mses)))

        best_idx = int(np.argmin(mean_cv_mse))
        best_lambda = float(lambda_grid[best_idx])

        # ------------------------------------------------------------------
        # Collect OOS residuals at the best lambda for sigma_hat estimation.
        # One pass through the folds collecting held-out predictions.
        # ------------------------------------------------------------------
        oos_residuals: list[np.ndarray] = []
        for train_idx, val_idx in folds:
            X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

            m = cls(
                n_estimators=n_estimators,
                lambda_=best_lambda,
                max_bins=max_bins,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            m.fit(X_tr, y_tr)
            preds_va = m.predict(X_va)
            oos_residuals.append(y_va - preds_va)

        oos_res_all = np.concatenate(oos_residuals)
        sigma_oos = float(np.sqrt(np.mean(oos_res_all ** 2)))

        # ------------------------------------------------------------------
        # Final fit on full dataset with best lambda
        # ------------------------------------------------------------------
        # Pass original X (may be polars) to preserve column names
        final_model = cls(
            n_estimators=n_estimators,
            lambda_=best_lambda,
            max_bins=max_bins,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        final_model.fit(X, y)

        # Store CV metadata
        final_model.cv_results_ = {
            "lambda_values": [float(lam) for lam in lambda_grid],
            "mean_cv_mse": mean_cv_mse,
        }

        # Override sigma_hat_ with the OOS estimate — it is less biased.
        # The training-based estimate is still accessible via the internal
        # computation in fit(), but we prefer the OOS version here.
        final_model.sigma_hat_oos_ = sigma_oos
        final_model.sigma_hat_ = sigma_oos

        return final_model

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: Union[np.ndarray, "pl.DataFrame"]) -> np.ndarray:
        """
        Predict the additive model for new samples.

        Parameters
        ----------
        X : np.ndarray or polars.DataFrame
            Feature matrix with the same columns (in the same order) as the
            training data.

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted values on the MSE scale.
        """
        self._check_fitted()
        X_arr, _ = _to_numpy_2d(X)
        n = X_arr.shape[0]
        out = np.full(n, self.intercept_)
        for k, (edges, scores) in enumerate(zip(self.bins_, self.shape_scores_)):
            idx = _digitize(X_arr[:, k], edges)
            out += scores[idx]
        return out

    # ------------------------------------------------------------------
    # Shape function accessor
    # ------------------------------------------------------------------

    def shape_function(self, feature: Union[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the fitted shape function for a feature.

        Parameters
        ----------
        feature : str or int
            Feature name (if trained on a polars DataFrame) or integer index.

        Returns
        -------
        bin_edges : np.ndarray
            Left bin edges (the last edge is +inf and is typically excluded
            from plots; use bin_edges[:-1] for the left boundaries).
        scores : np.ndarray
            Shape function values, one per bin.
        """
        self._check_fitted()
        k = self._feature_index(feature)
        return self.bins_[k], self.shape_scores_[k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BoulevardEBM has not been fitted. Call fit() first.")

    def _feature_index(self, feature: Union[str, int]) -> int:
        if isinstance(feature, (int, np.integer)):
            return int(feature)
        try:
            return self.feature_names_.index(feature)
        except ValueError:
            raise ValueError(
                f"Feature '{feature}' not found. "
                f"Available features: {self.feature_names_}"
            )

    def __repr__(self) -> str:
        fitted = self._fitted
        return (
            f"BoulevardEBM("
            f"n_estimators={self.n_estimators}, "
            f"lambda_={self.lambda_}, "
            f"max_bins={self.max_bins}, "
            f"fitted={fitted})"
        )


# ---------------------------------------------------------------------------
# BoulevardInference
# ---------------------------------------------------------------------------

class BoulevardInference:
    """
    Analytical CLT-based confidence intervals for BoulevardEBM shape functions.

    Wraps a fitted BoulevardEBM and computes bin-space influence norms that
    determine the pointwise standard error of each shape function estimate.

    Theory (Fang et al., Section 3 and Appendix B)
    -----------------------------------------------
    The Boulevard estimator converges in distribution to a kernel ridge
    regression (KRR) estimator.  For depth-1 stumps with equal-frequency
    binning, the kernel in bin space is:

        K^(k)_{jj'} = (1/n) * (n_j if j == j' else 0)   [diagonal]

    That is, the kernel matrix is proportional to the bin-frequency matrix H
    where H_{jj} = n_j / n.  The KRR solution in bin space satisfies:

        (H + lambda * J * H)^{-1} H y_bin  =  (I + lambda * J)^{-1} y_bin

    where J = I - (1/m) * 1*1^T is the centring matrix (removing the mean,
    since we explicitly centre each stump contribution).

    The influence function for predicting at bin j of feature k is:

        r^(k)(j) = e_j^T (I + lambda * J)^{-1} e_j / n_j

    where e_j is the j-th standard basis vector in R^m.  The pointwise SE is:

        SE^(k)(j) = sigma_hat * sqrt( r^(k)(j) / n )

    All matrices are at most 256 × 256, so inversion is O(m³) ≈ O(256³)
    per feature — a few milliseconds total.

    When the model was fitted via cv_lambda(), sigma_hat_ reflects the OOS
    residual standard deviation, which is less biased than the training
    residual version.  CIs will be narrower and better calibrated in that
    case.

    Parameters
    ----------
    model : BoulevardEBM
        A fitted BoulevardEBM instance.
    """

    def __init__(self, model: BoulevardEBM) -> None:
        if not model._fitted:
            raise RuntimeError(
                "BoulevardInference requires a fitted BoulevardEBM. "
                "Call model.fit() first."
            )
        self.model = model
        # Pre-compute per-feature influence norms
        self._influence_norms: list[np.ndarray] = self._compute_influence_norms()

    # ------------------------------------------------------------------
    # Influence norm computation
    # ------------------------------------------------------------------

    def _compute_influence_norms(self) -> list[np.ndarray]:
        """
        Compute the pointwise influence norm for every bin of every feature.

        Returns a list of length p, where element k is an array of length
        n_bins_k containing ||r^(k)(j)||^2 for each bin j.
        """
        norms = []
        n = self.model.n_samples_
        lam = self.model.lambda_

        for k in range(len(self.model.feature_names_)):
            counts = self.model.bin_counts_[k]
            m = len(counts)

            # Bin frequencies: h_j = n_j / n
            h = counts / n  # shape (m,)

            # Centring matrix J = I - (1/m) * 11^T
            J = np.eye(m) - np.ones((m, m)) / m

            # Kernel operator in bin space: A = I + lambda * J
            # (We work with A rather than (H + lambda*J*H) because the
            # equal-frequency binning makes H ≈ (1/m)*I up to fluctuations;
            # more precisely, after centring each stump the effective kernel
            # is (I + lambda*J), see Appendix B, Eq. 14-15.)
            A = np.eye(m) + lam * J  # shape (m, m)

            # Invert A.  It is symmetric positive definite for lambda > 0.
            try:
                A_inv = linalg.solve(A, np.eye(m), assume_a="pos")
            except linalg.LinAlgError:
                # Fallback: pinv for degenerate cases (very few bins)
                A_inv = np.linalg.pinv(A)

            # Influence norm for bin j:
            # ||r^(k)(j)||^2 = (A_inv[j,j])^2 / h_j
            # (from the hat-matrix diagonal; see Fang et al. Lemma 2)
            # We use h_j > 0 guard to avoid division by zero for empty bins.
            diag_A_inv = np.diag(A_inv)  # shape (m,)
            safe_h = np.where(h > 0, h, np.nan)
            # SE^2 = sigma^2 * diag_A_inv^2 / (n * h_j)
            # We store diag_A_inv^2 / h_j as the norm (sigma^2/n factor
            # applied in shape_ci)
            influence = np.where(h > 0, diag_A_inv ** 2 / safe_h, 0.0)
            norms.append(influence)

        return norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shape_ci(
        self,
        feature: Union[str, int],
        alpha: float = 0.05,
    ) -> pl.DataFrame:
        """
        Compute pointwise confidence intervals for a shape function.

        Parameters
        ----------
        feature : str or int
            Feature name or index.
        alpha : float
            Significance level.  Default 0.05 gives 95% CIs.

        Returns
        -------
        polars.DataFrame with columns:
            bin_label   : str   — "edge_left to edge_right" for each bin
            score       : float — fitted shape function value
            lower       : float — lower confidence bound
            upper       : float — upper confidence bound
            se          : float — pointwise standard error
        """
        from scipy.stats import norm as _norm

        self.model._check_fitted()
        k = self.model._feature_index(feature)
        edges = self.model.bins_[k]
        scores = self.model.shape_scores_[k]
        norms = self._influence_norms[k]

        z = float(_norm.ppf(1.0 - alpha / 2.0))
        se = self.model.sigma_hat_ * np.sqrt(norms / self.model.n_samples_)

        lower = scores - z * se
        upper = scores + z * se

        # Build human-readable bin labels
        n_bins = len(scores)
        labels = []
        for j in range(n_bins):
            left = edges[j]
            right = edges[j + 1]
            if np.isinf(right):
                labels.append(f"{left:.4g}+")
            else:
                labels.append(f"{left:.4g} – {right:.4g}")

        return pl.DataFrame({
            "bin_label": labels,
            "score": scores.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "se": se.tolist(),
        })

    def prediction_interval(
        self,
        X: Union[np.ndarray, "pl.DataFrame"],
        alpha: float = 0.05,
    ) -> pl.DataFrame:
        """
        Compute approximate prediction intervals for new observations.

        This is a naive marginal PI: prediction ± z * sigma_hat.  It does
        NOT account for estimation uncertainty in the shape functions (those
        are smaller than sigma_hat for typical insurance datasets).  For
        formal simultaneous bands use shape_ci on each feature separately.

        Parameters
        ----------
        X : np.ndarray or polars.DataFrame
        alpha : float

        Returns
        -------
        polars.DataFrame with columns: lower, prediction, upper
        """
        from scipy.stats import norm as _norm

        self.model._check_fitted()
        preds = self.model.predict(X)
        z = float(_norm.ppf(1.0 - alpha / 2.0))
        sigma = self.model.sigma_hat_

        return pl.DataFrame({
            "lower": (preds - z * sigma).tolist(),
            "prediction": preds.tolist(),
            "upper": (preds + z * sigma).tolist(),
        })

    def plot_shape(
        self,
        feature: Union[str, int],
        alpha: float = 0.05,
        ax: Optional[object] = None,
        show_data_density: bool = True,
    ) -> object:
        """
        Plot a shape function with CLT confidence band.

        Parameters
        ----------
        feature : str or int
            Feature name or index.
        alpha : float
            CI significance level.  Default 0.05 = 95% band.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.  A new figure is created if None.
        show_data_density : bool
            If True, add a histogram of training data density on a twin
            y-axis (right).  Useful for communicating where the CI is
            narrow (plenty of data) vs. wide (sparse bins).

        Returns
        -------
        matplotlib.axes.Axes
            The primary axes object.
        """
        import matplotlib.pyplot as plt

        self.model._check_fitted()
        k = self.model._feature_index(feature)
        fname = self.model.feature_names_[k]

        ci_df = self.shape_ci(feature, alpha=alpha)
        scores = ci_df["score"].to_numpy()
        lower = ci_df["lower"].to_numpy()
        upper = ci_df["upper"].to_numpy()

        edges = self.model.bins_[k]
        n_bins = len(scores)
        # x positions: left edge of each bin (exclude the +inf final edge)
        x = edges[:n_bins]
        # Use bin midpoints for line plot (approximate; last bin uses left edge)
        x_mid = np.array([
            (edges[j] + edges[j + 1]) / 2.0 if not np.isinf(edges[j + 1]) else edges[j]
            for j in range(n_bins)
        ])

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(x_mid, scores, color="steelblue", linewidth=1.8, label="Shape function")
        ax.fill_between(
            x_mid, lower, upper,
            alpha=0.25, color="steelblue",
            label=f"{int((1-alpha)*100)}% CI",
        )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(fname)
        ax.set_ylabel("Shape score")
        ax.set_title(f"Boulevard shape function: {fname}")
        ax.legend(loc="best", fontsize=9)

        if show_data_density:
            counts = self.model.bin_counts_[k]
            ax2 = ax.twinx()
            ax2.bar(
                x_mid, counts,
                width=np.diff(np.append(x, x[-1] * 1.05 if x[-1] != 0 else 1.0)),
                color="lightgray", alpha=0.4, label="Bin count",
            )
            ax2.set_ylabel("Bin count", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

        return ax

    def significance_table(self) -> pl.DataFrame:
        """
        Summary table of feature-level significance.

        For each feature reports:
        - max_absolute_score : max |score| across bins
        - ci_excludes_zero   : True if at least one bin's CI excludes zero
        - max_se             : maximum pointwise SE across bins

        Returns
        -------
        polars.DataFrame with one row per feature and columns:
            feature, max_absolute_score, ci_excludes_zero, max_se
        """
        self.model._check_fitted()
        rows = []
        for k, fname in enumerate(self.model.feature_names_):
            ci_df = self.shape_ci(k)
            scores = ci_df["score"].to_numpy()
            lower = ci_df["lower"].to_numpy()
            upper = ci_df["upper"].to_numpy()
            se = ci_df["se"].to_numpy()

            max_abs = float(np.max(np.abs(scores)))
            # CI excludes zero if any bin has lower > 0 or upper < 0
            excludes_zero = bool(np.any(lower > 0) or np.any(upper < 0))
            max_se = float(np.max(se))

            rows.append({
                "feature": fname,
                "max_absolute_score": max_abs,
                "ci_excludes_zero": excludes_zero,
                "max_se": max_se,
            })

        return pl.DataFrame(rows)
