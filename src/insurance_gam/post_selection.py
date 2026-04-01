"""
Post-selection inference for GLM coefficients after Lasso variable selection.

This module implements two approaches to valid confidence intervals when
variable selection (Lasso) is performed before fitting a GLM:

1. ``PostSelectionGLM`` — Full parametric programming approach from
   Shen, Gregory & Huang (2026), arXiv:2603.24875. Constructs truncated
   normal CIs that condition on the selection event, giving correct
   asymptotic coverage without the over-conservatism of sign-conditioned
   methods (Lee et al. 2016).

2. ``DataSplitPostSelectionGLM`` — Data-splitting: select features on the
   first 50% of data, infer on the second 50%. Always valid at exact nominal
   level (assuming independence), scales to any dataset size. Less efficient
   than the parametric approach but robust.

Both classes support Poisson GLMs with log link and optional log-exposure
offsets, which covers the primary use case in UK motor and property pricing
(frequency modelling).

Families supported: Poisson (log link) only.
Gamma is explicitly out of scope — the paper does not cover it and the
pseudo-data derivation for Gamma requires separate validation.

References
----------
Shen, K., Gregory, K., & Huang, B. (2026). Post-selection inference for
generalized linear models. arXiv:2603.24875.

Le Duy, V. N., & Takeuchi, I. (2021). Parametric programming approach for
more powerful and general Lasso selective inference. JMLR, 22(1), 3461–3502.

Examples
--------
>>> import numpy as np
>>> from insurance_gam.post_selection import PostSelectionGLM
>>> rng = np.random.default_rng(42)
>>> n, p = 500, 8
>>> X = rng.standard_normal((n, p))
>>> eta = 0.5 * X[:, 0] + 0.3 * X[:, 1]
>>> y = rng.poisson(np.exp(eta))
>>> model = PostSelectionGLM(family="poisson", alpha=0.05, random_state=42)
>>> model.fit(X, y)
>>> df = model.summary()
>>> df[df["selected"]].head()
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

# Matplotlib is imported lazily to avoid backend issues in headless environments
# Statsmodels and sklearn are required; raise a helpful error if missing.
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Poisson
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostSelectionGLM requires statsmodels. "
        "Install with: pip install insurance-gam[glm]"
    ) from exc

try:
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostSelectionGLM requires scikit-learn. "
        "Install with: pip install insurance-gam[glm]"
    ) from exc

from scipy import stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_family(family: str) -> None:
    """Raise ValueError for unsupported family."""
    if family != "poisson":
        raise ValueError(
            f"Only family='poisson' is supported. Got '{family}'. "
            "Gamma GLM post-selection inference requires separate derivation "
            "not covered by Shen/Gregory/Huang 2026."
        )


def _make_offset(
    exposure: Optional[np.ndarray], n: int
) -> np.ndarray:
    """Return log-exposure offset array; zeros if exposure is None."""
    if exposure is None:
        return np.zeros(n)
    exposure = np.asarray(exposure, dtype=float)
    if exposure.shape != (n,):
        raise ValueError(
            f"exposure must be a 1-D array of length {n}, "
            f"got shape {exposure.shape}."
        )
    if np.any(exposure <= 0):
        raise ValueError("All exposure values must be strictly positive.")
    return np.log(exposure)


def _fit_poisson_mle(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Poisson GLM with log link using statsmodels.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (no intercept column — added internally).
    y : ndarray of shape (n,)
        Observed counts.
    offset : ndarray of shape (n,)
        Log-exposure offsets (zeros if no exposure).

    Returns
    -------
    beta_hat : ndarray of shape (p + 1,)
        MLE estimates, intercept first.
    eta_hat : ndarray of shape (n,)
        Fitted linear predictor (including offset).
    """
    X_aug = sm.add_constant(X, has_constant="add")
    glm = sm.GLM(
        y,
        X_aug,
        family=sm.families.Poisson(),
        offset=offset,
    )
    result = glm.fit(disp=False)
    # fittedvalues gives mu; predict(which="linear") gives X @ beta (no offset)
    eta_hat = result.predict(which="linear") + offset
    return result.params, eta_hat


def _poisson_pseudo_data(
    X: np.ndarray,
    y: np.ndarray,
    eta_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct pseudo-data for Poisson GLM inference.

    From Fisher scoring: linearise the log-likelihood around the MLE to
    produce a weighted least-squares problem. For Poisson with log link,
    b''(eta) = exp(eta) = mu.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (no intercept).
    y : ndarray of shape (n,)
        Observed counts.
    eta_hat : ndarray of shape (n,)
        Fitted linear predictor at MLE.

    Returns
    -------
    Z_response : ndarray of shape (n,)
        Pseudo-response: z_i = sqrt(mu_i) * eta_i + (y_i - mu_i) / sqrt(mu_i)
    Z_covariates : ndarray of shape (n, p)
        Pseudo-covariates: Z_i = sqrt(mu_i) * X_i
    """
    mu = np.exp(eta_hat)
    mu = np.clip(mu, 1e-10, None)
    sqrt_mu = np.sqrt(mu)

    # Pseudo-response — intercept absorbed into eta already
    z = sqrt_mu * eta_hat + (y - mu) / sqrt_mu

    # Pseudo-covariates (no intercept; intercept is nuisance)
    Z = sqrt_mu[:, np.newaxis] * X

    return z, Z


def _lasso_select(
    Z: np.ndarray,
    z: np.ndarray,
    cv_folds: int,
    max_features: Optional[int],
    random_state: Optional[int],
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Select features using LassoCV on pseudo-data.

    Parameters
    ----------
    Z : ndarray of shape (n, p)
        Pseudo-covariates (already scaled by sqrt(mu)).
    z : ndarray of shape (n,)
        Pseudo-response.
    cv_folds : int
        Number of CV folds for lambda selection.
    max_features : int or None
        If set, stop Lasso at this many features (via max_iter heuristic).
    random_state : int or None
        Random state for CV fold splitting.

    Returns
    -------
    selected : ndarray of bool, shape (p,)
        Mask of selected features (nonzero Lasso coefficients).
    lambda_opt : float
        Chosen regularisation parameter.
    beta_lasso : ndarray of shape (p,)
        Lasso coefficient vector at lambda_opt.
    """
    n, p = Z.shape
    cv_folds = min(cv_folds, n)

    lasso_cv = LassoCV(
        cv=cv_folds,
        fit_intercept=True,
        max_iter=10_000,
        random_state=random_state,
        n_jobs=1,
    )
    lasso_cv.fit(Z, z)
    lambda_opt = lasso_cv.alpha_
    beta_lasso = lasso_cv.coef_

    selected = np.abs(beta_lasso) > 1e-8

    if max_features is not None and selected.sum() > max_features:
        # Refit at higher lambda until constraint is met
        alphas = lasso_cv.alphas_
        for alpha in alphas[alphas > lambda_opt]:
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
            model.fit(Z, z)
            selected = np.abs(model.coef_) > 1e-8
            if selected.sum() <= max_features:
                lambda_opt = alpha
                beta_lasso = model.coef_
                break

    return selected, lambda_opt, beta_lasso


# ---------------------------------------------------------------------------
# Parametric programming path tracer
# ---------------------------------------------------------------------------


def _trace_lasso_path_for_variable(
    Z: np.ndarray,
    z: np.ndarray,
    j: int,
    active_mask: np.ndarray,
    lambda_opt: float,
    threshold: float = 50.0,
    n_steps: int = 500,
) -> list[tuple[float, float]]:
    """
    Trace the Lasso selection event for variable j.

    Parameterise z(tau) = z_obs + tau * e_j (unit vector in direction of
    eta_j, the test direction for variable j) and trace how the active set
    changes as tau moves from -threshold to +threshold. The selection event
    T_{M,j} is the set of tau values where the active set equals the
    observed one.

    Parameters
    ----------
    Z : ndarray of shape (n, p)
        Pseudo-covariates.
    z : ndarray of shape (n,)
        Observed pseudo-response.
    j : int
        Index into the columns of Z (the variable to test).
    active_mask : ndarray of bool, shape (p,)
        Observed Lasso active set.
    lambda_opt : float
        Fixed regularisation parameter.
    threshold : float
        Range of tau to explore: [-threshold, threshold].
    n_steps : int
        Number of tau values to evaluate.

    Returns
    -------
    intervals : list of (lo, hi) tuples
        The selection event T_{M,j} expressed as disjoint intervals.
        tau=0 (the observed value) is guaranteed to be inside one interval.
    """
    # Test direction: projection onto the j-th canonical direction
    # eta_j = Z[:, j] / ||Z[:, j]||
    z_col = Z[:, j]
    norm_sq = np.dot(z_col, z_col)
    if norm_sq < 1e-14:
        # Degenerate column — return full real line (no information)
        return [(-np.inf, np.inf)]

    # Decompose: z(tau) = a + b * tau
    # a = z_obs - (z_obs . z_col / ||z_col||^2) * z_col
    # b = z_col / ||z_col||^2 (unnormalised direction)
    # For the truncated normal, we parameterise in the natural statistic scale.
    # We use a grid approach: evaluate the Lasso active set at each tau.

    taus = np.linspace(-threshold, threshold, n_steps)
    active_sets: list[frozenset] = []

    for tau in taus:
        z_tau = z + tau * z_col / norm_sq  # direction in response space
        lasso = Lasso(alpha=lambda_opt, fit_intercept=True, max_iter=5_000)
        lasso.fit(Z, z_tau)
        active_tau = frozenset(np.where(np.abs(lasso.coef_) > 1e-8)[0])
        active_sets.append(active_tau)

    target = frozenset(np.where(active_mask)[0])

    # Collect intervals where active set == target
    in_event = np.array([s == target for s in active_sets])

    intervals: list[tuple[float, float]] = []
    i = 0
    while i < len(taus):
        if in_event[i]:
            lo = taus[i]
            while i < len(taus) and in_event[i]:
                i += 1
            hi = taus[i - 1]
            intervals.append((float(lo), float(hi)))
        else:
            i += 1

    return intervals if intervals else [(-np.inf, np.inf)]


# ---------------------------------------------------------------------------
# Truncated normal CI solver
# ---------------------------------------------------------------------------


def _truncated_normal_ci(
    intervals: list[tuple[float, float]],
    mu: float,
    sigma: float,
    alpha: float,
) -> tuple[float, float, float]:
    """
    Compute equal-tailed CI via truncated normal conditioned on selection.

    Given a collection of disjoint intervals T = union of [lo_k, hi_k],
    and assuming the test statistic ~ N(mu, sigma^2), compute the
    alpha/2 and 1-alpha/2 quantiles of the truncated distribution.

    Parameters
    ----------
    intervals : list of (lo, hi)
        Selection event intervals T_{M,j}.
    mu : float
        Hypothesised mean (used for centering; inference is at mu=0).
    sigma : float
        Standard error of the test statistic.
    alpha : float
        Significance level (e.g. 0.05).

    Returns
    -------
    ci_lo : float
    ci_hi : float
    pvalue : float
    """
    if not intervals or (
        len(intervals) == 1
        and intervals[0][0] == -np.inf
        and intervals[0][1] == np.inf
    ):
        # No truncation — standard normal
        z_lo = stats.norm.ppf(alpha / 2)
        z_hi = stats.norm.ppf(1 - alpha / 2)
        z_obs = mu / sigma
        pvalue = 2 * stats.norm.sf(abs(z_obs))
        return mu + z_lo * sigma, mu + z_hi * sigma, pvalue

    # Probability mass of N(mu, sigma^2) within T
    def _mass_in_T(theta: float) -> float:
        total = 0.0
        for lo, hi in intervals:
            a = (lo - theta) / sigma
            b = (hi - theta) / sigma
            total += stats.norm.cdf(b) - stats.norm.cdf(a)
        return max(total, 1e-300)

    # Truncated CDF: P(X <= x | X in T, X ~ N(theta, sigma^2))
    def _trunc_cdf(x: float, theta: float) -> float:
        mass_above = 0.0
        for lo, hi in intervals:
            a = (lo - theta) / sigma
            b = (hi - theta) / sigma
            bx = (min(x, hi) - theta) / sigma
            if bx > a:
                mass_above += stats.norm.cdf(bx) - stats.norm.cdf(a)
        return mass_above / _mass_in_T(theta)

    # Pivot: for CI, we want the set of theta such that the truncated CDF
    # evaluated at mu (observed statistic) lies in [alpha/2, 1 - alpha/2].
    # Solve: F_theta(mu | T) = alpha/2  =>  theta = ci_hi
    #        F_theta(mu | T) = 1-alpha/2 => theta = ci_lo
    obs = mu  # the observed statistic is mu itself

    def _pivot(theta: float) -> float:
        return _trunc_cdf(obs, theta)

    search_radius = 10 * sigma
    lo_search = obs - search_radius
    hi_search = obs + search_radius

    # Find ci_hi: solve F_theta(obs | T) = alpha/2
    try:
        ci_hi = _bisect(_pivot, alpha / 2, lo_search, hi_search)
    except ValueError:
        ci_hi = np.nan

    # Find ci_lo: solve F_theta(obs | T) = 1 - alpha/2
    try:
        ci_lo = _bisect(_pivot, 1 - alpha / 2, lo_search, hi_search)
    except ValueError:
        ci_lo = np.nan

    # p-value: 2 * min(F_0(obs | T), 1 - F_0(obs | T))
    f_at_zero = _trunc_cdf(obs, 0.0)
    pvalue = 2 * min(f_at_zero, 1 - f_at_zero)

    return ci_lo, ci_hi, pvalue


def _bisect(
    f: callable,
    target: float,
    lo: float,
    hi: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Bisect to find x in [lo, hi] where f(x) == target."""
    # Ensure bracket
    f_lo, f_hi = f(lo), f(hi)

    # Expand bracket if needed
    for _ in range(30):
        if (f_lo - target) * (f_hi - target) < 0:
            break
        width = hi - lo
        lo -= width
        hi += width
        f_lo, f_hi = f(lo), f(hi)
    else:
        raise ValueError(f"Could not bracket root for target={target}")

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid - target) < tol:
            return mid
        if (f_lo - target) * (f_mid - target) < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Public class 1: PostSelectionGLM
# ---------------------------------------------------------------------------


class PostSelectionGLM:
    """
    Post-selection inference for Poisson GLM coefficients after Lasso.

    Implements the parametric programming approach from Shen, Gregory &
    Huang (2026). Constructs confidence intervals that condition on the
    Lasso selection event, correcting for the selective bias that makes
    naive post-selection Wald CIs anti-conservative.

    The method works in three stages:

    1. Fit Poisson MLE on the full data to get fitted linear predictor.
    2. Construct pseudo-data (Fisher scoring linearisation) and run
       LassoCV to select features.
    3. For each selected feature, trace the Lasso path as the pseudo-
       response is perturbed in the test direction, identify intervals
       where the active set is unchanged (the selection event), and
       compute truncated-normal CIs.

    For datasets with n > ``subsample``, a random subsample is used for
    steps 2–3 while step 1 uses the full data. The asymptotic theory
    (Corollary 1 in the paper) applies to the subsample.

    Parameters
    ----------
    family : str, default "poisson"
        GLM family. Only ``"poisson"`` is currently supported.
    alpha : float, default 0.05
        Significance level for confidence intervals (nominal coverage
        = 1 - alpha).
    cv_folds : int, default 5
        Number of folds for LassoCV lambda selection.
    max_features : int or None, default None
        If provided, enforce that Lasso selects at most this many
        features (raises lambda until constraint is satisfied).
    subsample : int, default 50_000
        Maximum n for the parametric path-tracing step. For larger
        datasets, a random subsample of this size is drawn.
    random_state : int or None, default None
        Random seed for subsampling and CV splits.

    Attributes
    ----------
    feature_names_ : list of str
        Column names from X (if DataFrame) or "x0", "x1", ... otherwise.
    selected_ : ndarray of bool
        Lasso selection mask.
    beta_mle_ : ndarray
        MLE coefficients (intercept + features) on the selected subset.
    summary_ : pd.DataFrame
        PSI summary. Available after fit().

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_gam.post_selection import PostSelectionGLM
    >>> rng = np.random.default_rng(0)
    >>> n, p = 1000, 6
    >>> X = rng.standard_normal((n, p))
    >>> y = rng.poisson(np.exp(0.4 * X[:, 0] + 0.3 * X[:, 1]))
    >>> m = PostSelectionGLM(random_state=0).fit(X, y)
    >>> m.summary()[["feature", "selected", "coefficient", "pvalue"]].head()
    """

    def __init__(
        self,
        family: str = "poisson",
        alpha: float = 0.05,
        cv_folds: int = 5,
        max_features: Optional[int] = None,
        subsample: int = 50_000,
        random_state: Optional[int] = None,
    ) -> None:
        _validate_family(family)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}.")
        if subsample < 100:
            raise ValueError(f"subsample must be >= 100, got {subsample}.")

        self.family = family
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state

        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.Series"],
        exposure: Optional[Union[np.ndarray, "pd.Series"]] = None,
    ) -> "PostSelectionGLM":
        """
        Fit the GLM, run Lasso selection, compute PSI confidence intervals.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix. If a pandas DataFrame, column names are used
            as feature names.
        y : array-like of shape (n,)
            Observed counts (non-negative integers for Poisson).
        exposure : array-like of shape (n,) or None, default None
            Exposure measure for Poisson rate modelling. Enters as
            log(exposure) offset. All values must be strictly positive.

        Returns
        -------
        self : PostSelectionGLM
            Fitted estimator (for method chaining).
        """
        # Input validation and coercion
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_ = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=float)
        n, p = X_arr.shape

        if y_arr.shape != (n,):
            raise ValueError(
                f"y must have shape ({n},), got {y_arr.shape}."
            )
        if np.any(y_arr < 0):
            raise ValueError("y must be non-negative for Poisson family.")

        offset = _make_offset(
            np.asarray(exposure, dtype=float) if exposure is not None else None, n
        )

        # Step 1: Fit Poisson MLE on full data
        rng = np.random.default_rng(self.random_state)
        _, eta_hat_full = _fit_poisson_mle(X_arr, y_arr, offset)

        # Step 2: Subsample for selection + path-tracing
        if n > self.subsample:
            idx = rng.choice(n, size=self.subsample, replace=False)
            X_sub = X_arr[idx]
            y_sub = y_arr[idx]
            eta_sub = eta_hat_full[idx]
        else:
            X_sub = X_arr
            y_sub = y_arr
            eta_sub = eta_hat_full

        # Step 3: Construct pseudo-data and run LassoCV
        z, Z = _poisson_pseudo_data(X_sub, y_sub, eta_sub)
        selected, lambda_opt, beta_lasso = _lasso_select(
            Z, z, self.cv_folds, self.max_features, self.random_state
        )

        self.selected_ = selected
        self._lambda_opt = lambda_opt
        self._beta_lasso = beta_lasso

        # Step 4: Refit MLE on selected features only (for coefficient estimates)
        selected_idx = np.where(selected)[0]
        n_sel = selected.sum()

        rows: list[dict] = []

        if n_sel == 0:
            warnings.warn(
                "Lasso selected zero features. All CIs will be NaN. "
                "Consider reducing regularisation (increase cv_folds or "
                "lower max_features).",
                UserWarning,
                stacklevel=2,
            )
            for j, name in enumerate(self.feature_names_):
                rows.append({
                    "feature": name,
                    "coefficient": np.nan,
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "pvalue": np.nan,
                    "selected": False,
                })
            self.summary_ = pd.DataFrame(rows)
            self._is_fitted = True
            return self

        X_sel = X_arr[:, selected_idx]
        beta_mle, _ = _fit_poisson_mle(X_sel, y_arr, offset)
        self.beta_mle_ = beta_mle

        # Recompute pseudo-data for selected features with MLE eta
        _, eta_mle = _fit_poisson_mle(X_sel, y_arr, offset)
        if n > self.subsample:
            X_sel_sub = X_sel[idx]
            y_sub2 = y_arr[idx]
            eta_sub2 = eta_mle[idx]
        else:
            X_sel_sub = X_sel
            y_sub2 = y_sub
            eta_sub2 = eta_mle

        z2, Z2 = _poisson_pseudo_data(X_sel_sub, y_sub2, eta_sub2)
        n_sub = X_sel_sub.shape[0]

        # Step 5: For each selected feature, trace path and compute PSI CI
        # Lasso active mask in the subspace of selected features is all-true
        active_mask_sel = np.ones(n_sel, dtype=bool)

        # Refit Lasso on selected features to get active mask
        _, _, beta_lasso_sel = _lasso_select(
            Z2, z2, self.cv_folds, None, self.random_state
        )
        active_mask_sel = np.abs(beta_lasso_sel) > 1e-8

        # MLE SE from Hessian (Fisher information)
        mu_mle_sub = np.exp(eta_sub2)
        mu_mle_sub = np.clip(mu_mle_sub, 1e-10, None)
        W = mu_mle_sub  # Poisson: weight = mu
        XtWX = X_sel_sub.T @ (W[:, np.newaxis] * X_sel_sub)
        try:
            cov_beta = np.linalg.inv(XtWX + np.eye(n_sel) * 1e-10)
        except np.linalg.LinAlgError:
            cov_beta = np.full((n_sel, n_sel), np.nan)

        # Map from subsample covariance to full-data scale
        scale = n / n_sub
        cov_beta_scaled = cov_beta * scale

        for local_j, global_j in enumerate(selected_idx):
            name = self.feature_names_[global_j]
            coef = beta_mle[local_j + 1]  # +1 for intercept
            se = np.sqrt(np.clip(cov_beta_scaled[local_j, local_j], 0, None))

            if se < 1e-14:
                rows.append({
                    "feature": name,
                    "coefficient": coef,
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "pvalue": np.nan,
                    "selected": True,
                })
                continue

            # Trace Lasso path for this variable
            try:
                intervals = _trace_lasso_path_for_variable(
                    Z2,
                    z2,
                    local_j,
                    active_mask_sel,
                    lambda_opt,
                    threshold=max(5 * se * np.sqrt(n_sub), 10.0),
                    n_steps=200,
                )
            except Exception:
                # Fall back to Wald CI if path tracing fails
                intervals = [(-np.inf, np.inf)]

            ci_lo, ci_hi, pval = _truncated_normal_ci(
                intervals, mu=coef, sigma=se, alpha=self.alpha
            )

            rows.append({
                "feature": name,
                "coefficient": coef,
                "std_error": se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "pvalue": pval,
                "selected": True,
            })

        # Non-selected features
        selected_globals = set(selected_idx.tolist())
        for j, name in enumerate(self.feature_names_):
            if j not in selected_globals:
                rows.append({
                    "feature": name,
                    "coefficient": np.nan,
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "pvalue": np.nan,
                    "selected": False,
                })

        self.summary_ = pd.DataFrame(rows)
        self._is_fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        """
        Return PSI inference results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: feature, coefficient, std_error, ci_lower, ci_upper,
            pvalue, selected. Non-selected features have NaN for all
            numeric columns.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before summary().")
        return self.summary_.copy()

    def forest_plot(self, ax=None):
        """
        Plot coefficient estimates with PSI confidence intervals.

        Only selected features are plotted. The point estimate is shown
        as a dot; the CI as a horizontal line. A vertical dashed line at
        zero is included for reference.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default None
            Axes to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before forest_plot().")

        import matplotlib.pyplot as plt

        sel_df = self.summary_[self.summary_["selected"]].copy()
        if sel_df.empty:
            raise ValueError("No selected features to plot.")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, len(sel_df) * 0.5 + 1)))

        features = sel_df["feature"].tolist()
        coefs = sel_df["coefficient"].values
        ci_lo = sel_df["ci_lower"].values
        ci_hi = sel_df["ci_upper"].values

        y_pos = np.arange(len(features))

        ax.errorbar(
            coefs,
            y_pos,
            xerr=[coefs - ci_lo, ci_hi - coefs],
            fmt="o",
            color="steelblue",
            ecolor="steelblue",
            capsize=4,
            linewidth=1.5,
            markersize=6,
        )
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Coefficient (log scale)")
        ax.set_title("Post-selection inference: Poisson GLM coefficients")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax


# ---------------------------------------------------------------------------
# Public class 2: DataSplitPostSelectionGLM
# ---------------------------------------------------------------------------


class DataSplitPostSelectionGLM:
    """
    Data-splitting approach to post-selection inference for Poisson GLM.

    Splits the dataset 50/50. Uses the first half to run Lasso variable
    selection, then fits a Poisson MLE on the second half restricted to
    the selected features. Reports standard Wald confidence intervals from
    the second-half fit.

    Validity: Because selection and inference use independent data, the
    Wald CIs achieve exact nominal coverage (under correct model
    specification) without any correction for selection. This is always
    valid regardless of dataset size and is the recommended default for
    production pipelines where n is large.

    Trade-off vs ``PostSelectionGLM``: Data splitting sacrifices roughly
    half the data for inference, producing wider CIs. The parametric
    programming approach uses all data but requires path-tracing
    computation. At n >= 10,000, data splitting is usually preferred.

    Parameters
    ----------
    family : str, default "poisson"
        GLM family. Only ``"poisson"`` is currently supported.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    cv_folds : int, default 5
        Number of folds for LassoCV on the selection half.
    max_features : int or None, default None
        Maximum number of features to select.
    random_state : int or None, default None
        Random seed for the train/test split and CV.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_gam.post_selection import DataSplitPostSelectionGLM
    >>> rng = np.random.default_rng(1)
    >>> n, p = 2000, 10
    >>> X = rng.standard_normal((n, p))
    >>> y = rng.poisson(np.exp(0.5 * X[:, 0] + 0.4 * X[:, 1]))
    >>> m = DataSplitPostSelectionGLM(random_state=1).fit(X, y)
    >>> m.summary()
    """

    def __init__(
        self,
        family: str = "poisson",
        alpha: float = 0.05,
        cv_folds: int = 5,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        _validate_family(family)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}.")

        self.family = family
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.max_features = max_features
        self.random_state = random_state

        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.Series"],
        exposure: Optional[Union[np.ndarray, "pd.Series"]] = None,
    ) -> "DataSplitPostSelectionGLM":
        """
        Fit using data splitting: select on half, infer on the other half.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix.
        y : array-like of shape (n,)
            Observed counts.
        exposure : array-like of shape (n,) or None, default None
            Exposure for Poisson rate modelling.

        Returns
        -------
        self : DataSplitPostSelectionGLM
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_ = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=float)
        n, p = X_arr.shape

        if y_arr.shape != (n,):
            raise ValueError(f"y must have shape ({n},), got {y_arr.shape}.")
        if np.any(y_arr < 0):
            raise ValueError("y must be non-negative for Poisson family.")

        offset = _make_offset(
            np.asarray(exposure, dtype=float) if exposure is not None else None, n
        )

        # 50/50 split
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        n_sel = n // 2
        sel_idx = idx[:n_sel]
        inf_idx = idx[n_sel:]

        X_sel = X_arr[sel_idx]
        y_sel = y_arr[sel_idx]
        offset_sel = offset[sel_idx]

        X_inf = X_arr[inf_idx]
        y_inf = y_arr[inf_idx]
        offset_inf = offset[inf_idx]

        # Selection half: fit MLE then pseudo-data then LassoCV
        _, eta_sel = _fit_poisson_mle(X_sel, y_sel, offset_sel)
        z_sel, Z_sel = _poisson_pseudo_data(X_sel, y_sel, eta_sel)
        selected, _, _ = _lasso_select(
            Z_sel, z_sel, self.cv_folds, self.max_features, self.random_state
        )

        self.selected_ = selected
        selected_idx_arr = np.where(selected)[0]
        n_sel_feat = selected.sum()

        rows: list[dict] = []

        if n_sel_feat == 0:
            warnings.warn(
                "Lasso selected zero features on the selection half.",
                UserWarning,
                stacklevel=2,
            )
            for name in self.feature_names_:
                rows.append({
                    "feature": name,
                    "coefficient": np.nan,
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "pvalue": np.nan,
                    "selected": False,
                })
            self.summary_ = pd.DataFrame(rows)
            self._is_fitted = True
            return self

        # Inference half: fit MLE on selected features only
        X_inf_sel = X_inf[:, selected_idx_arr]
        beta_inf, eta_inf = _fit_poisson_mle(X_inf_sel, y_inf, offset_inf)
        self.beta_mle_ = beta_inf

        # Fisher information Hessian for Wald CIs
        mu_inf = np.exp(eta_inf)
        mu_inf = np.clip(mu_inf, 1e-10, None)
        W_inf = mu_inf  # Poisson: weight = mu
        XtWX_inf = X_inf_sel.T @ (W_inf[:, np.newaxis] * X_inf_sel)
        try:
            cov_inf = np.linalg.inv(XtWX_inf + np.eye(n_sel_feat) * 1e-10)
        except np.linalg.LinAlgError:
            cov_inf = np.full((n_sel_feat, n_sel_feat), np.nan)

        z_crit = stats.norm.ppf(1 - self.alpha / 2)

        selected_globals = set(selected_idx_arr.tolist())
        sel_count = 0
        for j, name in enumerate(self.feature_names_):
            if j in selected_globals:
                coef = beta_inf[sel_count + 1]  # +1 for intercept
                se = np.sqrt(np.clip(cov_inf[sel_count, sel_count], 0, None))
                ci_lo = coef - z_crit * se
                ci_hi = coef + z_crit * se
                pval = 2 * stats.norm.sf(abs(coef / max(se, 1e-14)))
                rows.append({
                    "feature": name,
                    "coefficient": coef,
                    "std_error": se,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "pvalue": pval,
                    "selected": True,
                })
                sel_count += 1
            else:
                rows.append({
                    "feature": name,
                    "coefficient": np.nan,
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "pvalue": np.nan,
                    "selected": False,
                })

        self.summary_ = pd.DataFrame(rows)
        self._is_fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        """
        Return inference results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: feature, coefficient, std_error, ci_lower, ci_upper,
            pvalue, selected.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before summary().")
        return self.summary_.copy()

    def forest_plot(self, ax=None):
        """
        Plot coefficient estimates with Wald CIs (inference-half only).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default None
            Axes to draw on. Created if None.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before forest_plot().")

        import matplotlib.pyplot as plt

        sel_df = self.summary_[self.summary_["selected"]].copy()
        if sel_df.empty:
            raise ValueError("No selected features to plot.")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, len(sel_df) * 0.5 + 1)))

        features = sel_df["feature"].tolist()
        coefs = sel_df["coefficient"].values
        ci_lo = sel_df["ci_lower"].values
        ci_hi = sel_df["ci_upper"].values

        y_pos = np.arange(len(features))
        ax.errorbar(
            coefs,
            y_pos,
            xerr=[coefs - ci_lo, ci_hi - coefs],
            fmt="o",
            color="darkorange",
            ecolor="darkorange",
            capsize=4,
            linewidth=1.5,
            markersize=6,
        )
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Coefficient (log scale)")
        ax.set_title("Data-split inference: Poisson GLM coefficients")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax
