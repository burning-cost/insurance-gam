"""
Penalized GLM inference: fit once, then request CIs from either strategy.

This module is the user-facing complement to :mod:`insurance_gam.debiased_glm`.
Both implement Manna, Huang, Dey, Gu & He (2025), arXiv:2410.01008, but they
have different ergonomics for different workflows.

:class:`DebiasedGLM` commits to a CI strategy at construction time — it is the
right choice when you already know which strategy you want and want a single
``fit().summary()`` call.

:class:`PenalizedGLMInference` separates fitting from inference: you call
``fit()`` once to run the penalized GLM and store the sufficient statistics,
then call ``confidence_intervals()`` or ``bootstrap_ci()`` independently.
This is useful when:

- You are exploring which strategy produces more stable CIs for a given dataset.
- You want both asymptotic *and* bootstrap CIs in the same session without
  refitting the penalized model.
- You need to sweep confidence levels post-hoc without re-running the IRLS loop.

**Mathematical background:**

After Lasso/Elastic Net selects features with parameter lambda, the penalized
coefficient estimates beta_hat are biased toward zero.  For large true
coefficients, naive Wald intervals constructed around beta_hat have coverage
well below the nominal level.  Two corrections are available:

*Asymptotic (Strategy A)*: One-step Newton correction using the inverse Hessian::

    b_hat = beta_hat - Theta @ gradient_l(beta_hat)
    Theta = (X^T diag(W) X / n)^{-1}       # sample Hessian inverse
    gradient = X^T (mu_hat - y) / n         # gradient of neg log-likelihood

This produces CIs of the form ``b_hat +/- z * SE`` where SE = sqrt(diag(Theta)/n).
Valid when p^2/n -> 0 (Xia, Nan & Li 2021 Biometrics; Manna et al. 2025).

*Bootstrap (Strategy B)*: Pearson-residual parametric bootstrap::

    r_i = (y_i - mu_hat_i) / sqrt(V(mu_hat_i))   # Pearson residuals
    y_i^* = mu_hat_i + sqrt(V(mu_hat_i)) * r_i^* # resample r*
    Refit lasso at fixed lambda on each y^*, collect quantiles of beta_hat^*
    CI: [2*beta_hat - Q(1-alpha/2), 2*beta_hat - Q(alpha/2)]  (pivot form)

**Families supported:**

- ``"poisson"``: log link, V(mu) = mu, dispersion phi = 1 (fixed)
- ``"gamma"``: log link, V(mu) = mu^2 / phi, phi estimated by Pearson chi^2
- ``"tweedie"``: log link, V(mu) = mu^p / phi, p in (1, 2)

All three families are relevant to UK motor and property pricing:
Poisson for claim frequency, Gamma for claim severity, Tweedie for combined
loss cost.

**Relationship to other classes in this package:**

- :class:`PostSelectionGLM`: selective inference via parametric programming
  (conditions on the Lasso selection event). Answers "is this variable
  genuinely non-zero?" Poisson only.
- :class:`DebiasedGLM`: strategy fixed at construction; ``fit().summary()``
  API. Same math as this class.
- :class:`PenalizedGLMInference` (this class): fit once, then call either
  strategy separately. The exploratory interface.

References
----------
Manna, A., Huang, B., Dey, D. K., Gu, C., & He, X. (2025). Interval
estimation of coefficients in penalized regression models of insurance data.
Applied Stochastic Models in Business and Industry. arXiv:2410.01008.

Xia, L., Nan, B., & Li, Y. (2021). A revisit of the de-biased lasso in
generalized linear model. Biometrics, 77(4), 1468-1480.

Examples
--------
Fit a Poisson frequency model, then compare both CI strategies:

>>> import numpy as np
>>> from insurance_gam.penalized_glm_inference import PenalizedGLMInference
>>> rng = np.random.default_rng(0)
>>> n, p = 3000, 10
>>> X = rng.standard_normal((n, p))
>>> y = rng.poisson(np.exp(0.4 * X[:, 0] + 0.3 * X[:, 1]))
>>> m = PenalizedGLMInference(family="poisson").fit(X, y)
>>> df_asym = m.confidence_intervals(alpha=0.05)         # Strategy A
>>> df_boot = m.bootstrap_ci(alpha=0.05, n_bootstrap=200) # Strategy B
>>> m.summary()  # calls confidence_intervals() under the hood
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PenalizedGLMInference requires scikit-learn. "
        "Install with: pip install insurance-gam[glm]"
    ) from exc


# ---------------------------------------------------------------------------
# Supported families
# ---------------------------------------------------------------------------

_SUPPORTED_FAMILIES = ("poisson", "gamma", "tweedie")


def _check_family(family: str, tweedie_power: float) -> None:
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"family must be one of {_SUPPORTED_FAMILIES}, got '{family}'."
        )
    if family == "tweedie" and not (1.0 <= tweedie_power <= 2.0):
        raise ValueError(
            f"tweedie_power must be in [1, 2] for compound Poisson-Gamma, "
            f"got {tweedie_power}."
        )


# ---------------------------------------------------------------------------
# Family-specific computations
# ---------------------------------------------------------------------------


def _hessian_weights(
    mu: np.ndarray, family: str, phi: float, p: float
) -> np.ndarray:
    """
    Diagonal IRLS weights W_i = mu_i^{2-p} / phi for log-link GLMs.

    Derivation from the Tweedie negative log-likelihood::

        l_i = -y_i * exp((1-p)*eta_i)/(1-p) + exp((2-p)*eta_i)/(2-p)
        d^2 l_i / d eta_i^2 = exp((2-p)*eta_i) = mu_i^{2-p}

    With dispersion: W_i = mu_i^{2-p} / phi.
    Poisson (p=1): W_i = mu_i. Gamma (p=2): W_i = 1/phi.
    """
    if family == "poisson":
        return mu  # p=1, phi=1: mu^(2-1)/1 = mu
    elif family == "gamma":
        return np.ones_like(mu) / phi  # p=2: mu^0/phi = 1/phi
    else:  # tweedie
        return mu ** (2.0 - p) / phi


def _sqrt_variance(
    mu: np.ndarray, family: str, phi: float, p: float
) -> np.ndarray:
    """
    sqrt(V(mu)) for computing Pearson residuals.

    V(mu) = mu^p / phi for Tweedie family.
    Poisson: sqrt(mu). Gamma: mu/sqrt(phi). Tweedie: mu^{p/2} * sqrt(phi).
    """
    if family == "poisson":
        return np.sqrt(mu)
    elif family == "gamma":
        return mu / np.sqrt(phi)
    else:  # tweedie
        return mu ** (p / 2.0) * np.sqrt(phi)


def _estimate_dispersion(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    p: float,
    n_params: int,
) -> float:
    """
    Pearson chi-squared dispersion estimate.

    phi_hat = sum((y - mu)^2 / V_raw(mu)) / (n - n_params)

    where V_raw omits phi (so phi can be estimated from this).
    Poisson: phi is fixed at 1.0 (not estimated).
    """
    if family == "poisson":
        return 1.0
    n = len(y)
    dof = max(n - n_params, 1)
    if family == "gamma":
        v_raw = mu ** 2
    else:
        v_raw = mu ** p
    pearson_chi2 = np.sum((y - mu) ** 2 / np.clip(v_raw, 1e-14, None))
    return float(pearson_chi2 / dof)


def _make_offset(exposure: Optional[np.ndarray], n: int) -> np.ndarray:
    """Return log(exposure) offset; zeros if exposure is None."""
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


# ---------------------------------------------------------------------------
# Penalized GLM fitting via IRLS + ElasticNet
# ---------------------------------------------------------------------------


def _fit_penalized_irls(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    family: str,
    phi: float,
    tweedie_power: float,
    alpha: float,
    l1_ratio: float,
    cv_folds: int,
    random_state: Optional[int],
    scaler: StandardScaler,
    fit_scaler: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    IRLS-based penalized GLM fit.

    Iteratively re-weighted least squares: at each step we linearise the
    GLM log-likelihood to construct a weighted pseudo-response, then fit
    ElasticNet on the weighted problem.  This is Fisher scoring for the
    penalized problem.

    For log-link families the working response is::

        z_i = eta_i + (y_i - mu_i) / mu_i

    which is the first-order Taylor expansion of the deviance gradient.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    offset : ndarray (n,)
    family : str
    phi : float
    tweedie_power : float
    alpha : float
        ElasticNet alpha. If <= 0, use CV.
    l1_ratio : float
    cv_folds : int
    random_state : int or None
    scaler : StandardScaler
    fit_scaler : bool

    Returns
    -------
    beta_hat : ndarray (p,)
        Coefficient vector in original X scale.
    mu_hat : ndarray (n,)
        Fitted means.
    intercept : float
    lambda_opt : float
        ElasticNet regularization strength used.
    """
    n, p = X.shape

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Starting values: log(y + 0.1) as initial linear predictor
    eta = np.log(np.clip(y, 0.1, None)) - offset

    max_irls = 20
    tol_irls = 1e-6
    beta_hat = np.zeros(p)
    intercept = np.mean(eta)
    lambda_opt = alpha

    for irls_iter in range(max_irls):
        mu = np.exp(np.clip(eta + offset, -30, 30))
        mu = np.clip(mu, 1e-10, None)

        W = _hessian_weights(mu, family, phi, tweedie_power)
        W = np.clip(W, 1e-14, None)
        sqrt_W = np.sqrt(W)

        # Working response for log link
        z = eta + (y - mu) / mu
        z_weighted = sqrt_W * z
        X_weighted = sqrt_W[:, np.newaxis] * X_scaled

        if alpha > 0:
            en = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=10_000,
                warm_start=False,
            )
            en.fit(X_weighted, z_weighted)
        else:
            if irls_iter == 0:
                en_cv = ElasticNetCV(
                    l1_ratio=l1_ratio,
                    cv=cv_folds,
                    fit_intercept=True,
                    max_iter=10_000,
                    n_jobs=1,
                    random_state=random_state,
                )
                en_cv.fit(X_weighted, z_weighted)
                lambda_opt = en_cv.alpha_
                en = en_cv
            else:
                en = ElasticNet(
                    alpha=lambda_opt,
                    l1_ratio=l1_ratio,
                    fit_intercept=True,
                    max_iter=10_000,
                )
                en.fit(X_weighted, z_weighted)

        beta_scaled = en.coef_
        scale = scaler.scale_
        beta_hat_new = beta_scaled / scale
        intercept_new = en.intercept_ - np.dot(scaler.mean_ / scale, beta_scaled)
        eta_new = X @ beta_hat_new + intercept_new

        delta = np.max(np.abs(beta_hat_new - beta_hat)) + abs(intercept_new - intercept)
        beta_hat = beta_hat_new
        intercept = intercept_new
        eta = eta_new

        if irls_iter > 0 and delta < tol_irls:
            break

    mu_hat = np.exp(np.clip(eta + offset, -30, 30))
    mu_hat = np.clip(mu_hat, 1e-10, None)

    return beta_hat, mu_hat, intercept, lambda_opt


# ---------------------------------------------------------------------------
# Strategy A: Asymptotic debiasing
# ---------------------------------------------------------------------------


def _asymptotic_ci(
    X: np.ndarray,
    y: np.ndarray,
    mu_hat: np.ndarray,
    beta_hat: np.ndarray,
    family: str,
    phi: float,
    tweedie_power: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-step Newton debiasing and asymptotic Wald CIs.

    Formula (Manna et al. 2025, following Xia et al. 2021)::

        b_hat = beta_hat - Theta @ gradient
        Theta = (X^T diag(W) X / n)^{-1}
        gradient = X^T (mu_hat - y) / n
        SE_j = sqrt(Theta_{jj} / n)
        CI_j = b_hat_j +/- z_{alpha/2} * SE_j

    The variance formula V = Theta/n uses the fact that for a correctly
    specified GLM, the expected Fisher information equals the sample Hessian
    in the limit.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    mu_hat : ndarray (n,)
    beta_hat : ndarray (p,)
    family, phi, tweedie_power : family specification
    alpha : float
        Significance level (not confidence — so alpha=0.05 for 95% CIs).

    Returns
    -------
    b_debiased : ndarray (p,)
    se : ndarray (p,)
    ci_lower : ndarray (p,)
    ci_upper : ndarray (p,)
    pvalues : ndarray (p,)
    """
    n, p = X.shape

    W = _hessian_weights(mu_hat, family, phi, tweedie_power)
    H = (X.T * W) @ X / n  # (p, p)

    # Warn if p^2/n is too large for the asymptotic guarantee
    ratio = p ** 2 / n
    if ratio > 0.25:
        warnings.warn(
            f"p^2/n = {ratio:.2f} > 0.25. The asymptotic debiased estimator "
            "requires p^2/n -> 0. With moderate p and small n, bootstrap_ci() "
            "may be more reliable.",
            UserWarning,
            stacklevel=5,
        )

    # Hessian inversion with ridge regularisation if near-singular
    try:
        cond = np.linalg.cond(H)
        if cond > 1e12:
            ridge = 1e-8 * np.trace(H) / p
            Theta = np.linalg.inv(H + np.eye(p) * ridge)
        else:
            Theta = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Hessian is singular; using pseudoinverse. Results may be unreliable.",
            UserWarning,
            stacklevel=5,
        )
        Theta = np.linalg.pinv(H)

    # Gradient of negative log-likelihood: grad_j = sum_i (mu_i - y_i) x_ij / n
    grad = X.T @ (mu_hat - y) / n

    # Debiased estimate
    b_debiased = beta_hat - Theta @ grad

    # SE from diagonal of Theta/n
    se = np.sqrt(np.clip(np.diag(Theta), 0, None) / n)

    z_crit = stats.norm.ppf(1.0 - alpha / 2)
    ci_lower = b_debiased - z_crit * se
    ci_upper = b_debiased + z_crit * se
    pvalues = 2 * stats.norm.sf(np.abs(b_debiased / np.clip(se, 1e-14, None)))

    return b_debiased, se, ci_lower, ci_upper, pvalues


# ---------------------------------------------------------------------------
# Strategy B: Pearson residual bootstrap
# ---------------------------------------------------------------------------


def _pearson_bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    mu_hat: np.ndarray,
    beta_hat: np.ndarray,
    family: str,
    phi: float,
    tweedie_power: float,
    alpha: float,
    l1_ratio: float,
    lambda_opt: float,
    n_bootstrap: int,
    random_state: Optional[int],
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pearson residual parametric bootstrap CIs (Algorithm 2, Manna et al. 2025).

    Steps:
    1. Compute Pearson residuals r_i = (y_i - mu_hat_i) / sqrt(V(mu_hat_i)).
    2. Mean-centre residuals.
    3. For b = 1 ... n_bootstrap:
       a. Resample (with replacement) from centred residuals.
       b. Reconstruct y^* = mu_hat + sqrt(V(mu_hat)) * r^*.
       c. Refit penalized GLM at fixed lambda_opt.
       d. Store beta_hat^*(b).
    4. Pivot CI: [2*beta_hat - Q(1-alpha/2), 2*beta_hat - Q(alpha/2)].

    The pivot form accounts for the shrinkage bias in the bootstrap distribution
    symmetrically around beta_hat, which is the key contribution for large
    coefficients.

    Parameters
    ----------
    X, y, offset, mu_hat, beta_hat : arrays as above
    family, phi, tweedie_power : family specification
    alpha : float
        Significance level.
    l1_ratio : float
    lambda_opt : float
        Fixed regularisation parameter from the original fit.
    n_bootstrap : int
    random_state : int or None
    scaler : fitted StandardScaler

    Returns
    -------
    ci_lower : ndarray (p,)
    ci_upper : ndarray (p,)
    """
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    sqrt_v = _sqrt_variance(mu_hat, family, phi, tweedie_power)
    sqrt_v = np.clip(sqrt_v, 1e-14, None)

    r = (y - mu_hat) / sqrt_v
    r_centred = r - r.mean()

    boot_coefs = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        r_boot = rng.choice(r_centred, size=n, replace=True)
        y_boot = mu_hat + sqrt_v * r_boot

        # Clip to valid support for family
        if family == "poisson":
            y_boot = np.clip(y_boot, 0.0, None)
        else:
            y_boot = np.clip(y_boot, 1e-10, None)

        try:
            beta_boot, _, _, _ = _fit_penalized_irls(
                X, y_boot, offset,
                family=family,
                phi=phi,
                tweedie_power=tweedie_power,
                alpha=lambda_opt,
                l1_ratio=l1_ratio,
                cv_folds=5,
                random_state=None,
                scaler=scaler,
                fit_scaler=False,
            )
        except Exception:
            beta_boot = beta_hat.copy()

        boot_coefs[b] = beta_boot

    # Pivot CI
    q_lo = np.percentile(boot_coefs, 100.0 * (1.0 - alpha / 2), axis=0)
    q_hi = np.percentile(boot_coefs, 100.0 * (alpha / 2), axis=0)
    ci_lower = 2.0 * beta_hat - q_lo
    ci_upper = 2.0 * beta_hat - q_hi

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Public class: PenalizedGLMInference
# ---------------------------------------------------------------------------


class PenalizedGLMInference:
    """
    Penalized GLM inference: fit once, then explore both CI strategies.

    Implements bias-corrected confidence intervals for Elastic Net / Lasso
    penalized GLM coefficients in the Tweedie family (Poisson, Gamma, Tweedie).
    Based on Manna et al. (2025), arXiv:2410.01008.

    Unlike :class:`DebiasedGLM`, this class separates the penalized fit from
    the CI computation.  Call ``fit()`` once to run IRLS and store sufficient
    statistics, then call ``confidence_intervals()`` for asymptotic CIs or
    ``bootstrap_ci()`` for Pearson residual bootstrap CIs — or both, to compare.

    Typical use case: after fitting, call ``confidence_intervals()`` to get
    quick asymptotic CIs, then call ``bootstrap_ci()`` on the same fitted object
    if you want to validate with a heavier bootstrap run.

    **When to use this class vs DebiasedGLM:**

    - Use :class:`PenalizedGLMInference` when you want to compare strategies
      or call both without re-fitting.
    - Use :class:`DebiasedGLM` when you know which strategy you want and prefer
      a single ``fit().summary()`` call.

    **When to use asymptotic vs bootstrap CIs:**

    - Large portfolios (n > 50,000, selected p < 100): asymptotic CIs are fast
      and well-justified.
    - Small portfolios (n < 10,000) or suspect model misspecification: bootstrap
      CIs are safer.
    - Both available: check agreement. Large discrepancies suggest either
      near-singular Hessian or the p^2/n condition is violated.

    Parameters
    ----------
    family : str, default "poisson"
        GLM family. One of ``"poisson"``, ``"gamma"``, ``"tweedie"``.
    tweedie_power : float, default 1.5
        Variance power for Tweedie: ``p=1`` is Poisson, ``p=2`` is Gamma.
        Ignored unless ``family="tweedie"``. Must be in [1, 2].
    alpha : float, default 1.0
        ElasticNet regularisation strength. If 0.0, use cross-validation
        (slower but automatic). Positive values use the given alpha directly.
    l1_ratio : float, default 0.5
        ElasticNet mixing: 1.0 = pure Lasso, 0.0 = Ridge. Values in (0, 1)
        give Elastic Net, which handles correlated rating factors better than
        pure Lasso.
    cv_folds : int, default 5
        CV folds for lambda selection when ``alpha=0.0``.
    random_state : int or None, default None
        Random seed for CV splits and bootstrap.

    Attributes
    ----------
    coef_penalized_ : ndarray of shape (n_features,)
        Penalized (biased) coefficient estimates from the IRLS fit.
    intercept_ : float
        Fitted intercept.
    selected_features_ : ndarray of bool, shape (n_features,)
        Features with non-zero penalized coefficient.
    feature_names_ : list of str
        Feature names (from DataFrame columns or ``"x0"``, ``"x1"``, ...).
    phi_ : float
        Estimated dispersion parameter.
    lambda_ : float
        ElasticNet regularisation parameter used.
    n_obs_ : int
        Number of observations used for fitting.
    n_features_ : int
        Number of input features.

    Examples
    --------
    Poisson frequency model:

    >>> import numpy as np
    >>> from insurance_gam.penalized_glm_inference import PenalizedGLMInference
    >>> rng = np.random.default_rng(0)
    >>> n, p = 3000, 10
    >>> X = rng.standard_normal((n, p))
    >>> y = rng.poisson(np.exp(0.4 * X[:, 0] + 0.3 * X[:, 1]))
    >>> m = PenalizedGLMInference(family="poisson").fit(X, y)
    >>> df = m.confidence_intervals(alpha=0.05)
    >>> df[df["selected"]].head()

    Compare asymptotic and bootstrap on the same fitted model:

    >>> df_asym = m.confidence_intervals(alpha=0.05)
    >>> df_boot = m.bootstrap_ci(alpha=0.05, n_bootstrap=500)
    """

    def __init__(
        self,
        family: str = "poisson",
        tweedie_power: float = 1.5,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        _check_family(family, tweedie_power)
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}.")

        self.family = family
        self.tweedie_power = tweedie_power
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.random_state = random_state

        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.Series"],
        exposure: Optional[Union[np.ndarray, "pd.Series"]] = None,
    ) -> "PenalizedGLMInference":
        """
        Fit the penalized GLM and store sufficient statistics for inference.

        This runs IRLS + ElasticNet (with optional CV for lambda), estimates
        dispersion phi (for Gamma/Tweedie), and stores everything needed for
        subsequent calls to :meth:`confidence_intervals` and :meth:`bootstrap_ci`.
        The penalized estimates themselves are stored in ``coef_penalized_``
        but are biased; use the CI methods to obtain corrected estimates.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix. If a pandas DataFrame, column names become feature
            names.
        y : array-like of shape (n,)
            Response variable. Must be non-negative for Poisson; strictly
            positive for Gamma/Tweedie.
        exposure : array-like of shape (n,) or None, default None
            Exposure for rate modelling. Enters as log(exposure) offset.
            All values must be strictly positive.

        Returns
        -------
        self : PenalizedGLMInference
            Fitted estimator.

        Raises
        ------
        ValueError
            If inputs fail validation (wrong shape, out-of-range values,
            negative response for Poisson, etc.).
        """
        # Input coercion
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
        if self.family == "poisson" and np.any(y_arr < 0):
            raise ValueError("y must be non-negative for Poisson family.")
        if self.family in ("gamma", "tweedie") and np.any(y_arr <= 0):
            raise ValueError("y must be strictly positive for Gamma/Tweedie family.")

        offset = _make_offset(
            np.asarray(exposure, dtype=float) if exposure is not None else None, n
        )

        # First IRLS pass
        scaler = StandardScaler()
        beta_hat, mu_hat, intercept, lambda_opt = _fit_penalized_irls(
            X_arr, y_arr, offset,
            family=self.family,
            phi=1.0,
            tweedie_power=self.tweedie_power,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            scaler=scaler,
            fit_scaler=True,
        )

        # Estimate dispersion from Pearson chi-squared
        phi = _estimate_dispersion(
            y_arr, mu_hat, self.family, self.tweedie_power, p + 1
        )

        # Re-fit with updated dispersion if it has changed meaningfully
        if self.family != "poisson" and abs(phi - 1.0) > 0.01:
            beta_hat, mu_hat, intercept, lambda_opt = _fit_penalized_irls(
                X_arr, y_arr, offset,
                family=self.family,
                phi=phi,
                tweedie_power=self.tweedie_power,
                alpha=lambda_opt,  # fixed from first pass
                l1_ratio=self.l1_ratio,
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                scaler=scaler,
                fit_scaler=False,
            )
            phi = _estimate_dispersion(
                y_arr, mu_hat, self.family, self.tweedie_power, p + 1
            )

        # Store sufficient statistics for inference
        self._X = X_arr
        self._y = y_arr
        self._offset = offset
        self._mu_hat = mu_hat
        self._scaler = scaler

        self.coef_penalized_ = beta_hat
        self.intercept_ = intercept
        self.selected_features_ = np.abs(beta_hat) > 1e-8
        self.phi_ = phi
        self.lambda_ = lambda_opt
        self.n_obs_ = n
        self.n_features_ = p

        self._is_fitted = True
        return self

    def confidence_intervals(
        self,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Asymptotic debiased confidence intervals (Strategy A).

        Applies the one-step Newton correction to produce debiased coefficient
        estimates and Wald-style confidence intervals with asymptotic nominal
        coverage.  Requires p^2/n -> 0 for validity (a warning is issued if
        the ratio exceeds 0.25).

        This method is fast (O(n * p^2) dominated by the Hessian inversion)
        and should be the default choice for large portfolios.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for CIs (not confidence level). 0.05 gives 95%
            CIs, 0.01 gives 99% CIs. Must be in (0, 1).

        Returns
        -------
        pd.DataFrame
            One row per feature. Columns:

            - ``feature``: feature name
            - ``coef_penalized``: raw penalized estimate (biased)
            - ``coef``: debiased estimate (bias-corrected)
            - ``se``: asymptotic standard error
            - ``ci_lower``: lower bound of confidence interval
            - ``ci_upper``: upper bound
            - ``pvalue``: two-sided p-value (H0: beta_j = 0)
            - ``selected``: bool, whether Lasso selected this feature

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If alpha is not in (0, 1).
        """
        self._check_fitted()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

        b_debiased, se, ci_lower, ci_upper, pvalues = _asymptotic_ci(
            self._X, self._y, self._mu_hat, self.coef_penalized_,
            self.family, self.phi_, self.tweedie_power, alpha,
        )

        return self._build_dataframe(
            coef_debiased=b_debiased,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalues=pvalues,
        )

    def bootstrap_ci(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 500,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pearson residual bootstrap confidence intervals (Strategy B).

        Constructs CIs by resampling Pearson residuals from the fitted model,
        reconstructing bootstrap responses, refitting the penalized GLM at
        the same lambda, and computing pivot-form quantile CIs.  More
        computationally expensive than :meth:`confidence_intervals` but valid
        under weaker assumptions (no p^2/n condition).

        The pivot CI corrects for bias in the penalized estimate::

            CI = [2*beta_hat - Q(1-alpha/2), 2*beta_hat - Q(alpha/2)]

        where Q(q) is the q-th quantile of the bootstrap distribution of
        beta_hat^*.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level. 0.05 gives 95% CIs.
        n_bootstrap : int, default 500
            Number of bootstrap resamples. 200 is fast for exploration;
            1000 for final results.
        random_state : int or None, default None
            Override the instance random_state for this call only. Useful
            for comparing multiple bootstrap runs on the same fitted model.

        Returns
        -------
        pd.DataFrame
            One row per feature. Columns:

            - ``feature``: feature name
            - ``coef_penalized``: raw penalized estimate (biased)
            - ``coef``: same as coef_penalized (bootstrap CIs use the penalized
              point estimate; bias is corrected in the CI bounds, not the center)
            - ``se``: NaN (bootstrap does not produce a standard error)
            - ``ci_lower``: lower bound
            - ``ci_upper``: upper bound
            - ``pvalue``: NaN (bootstrap does not produce a p-value)
            - ``selected``: bool

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If alpha is not in (0, 1) or n_bootstrap < 1.
        """
        self._check_fitted()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")

        seed = random_state if random_state is not None else self.random_state

        ci_lower, ci_upper = _pearson_bootstrap_ci(
            self._X, self._y, self._offset, self._mu_hat, self.coef_penalized_,
            family=self.family,
            phi=self.phi_,
            tweedie_power=self.tweedie_power,
            alpha=alpha,
            l1_ratio=self.l1_ratio,
            lambda_opt=self.lambda_,
            n_bootstrap=n_bootstrap,
            random_state=seed,
            scaler=self._scaler,
        )

        p = self.n_features_
        return self._build_dataframe(
            coef_debiased=self.coef_penalized_,  # bootstrap: point est = penalized
            se=np.full(p, np.nan),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalues=np.full(p, np.nan),
        )

    def summary(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return a summary DataFrame using asymptotic CIs (Strategy A).

        Convenience method: equivalent to ``confidence_intervals(alpha=alpha)``.
        For bootstrap CIs, call :meth:`bootstrap_ci` directly.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        pd.DataFrame
            Same format as :meth:`confidence_intervals`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        return self.confidence_intervals(alpha=alpha)

    def forest_plot(
        self,
        alpha: float = 0.05,
        strategy: str = "asymptotic",
        n_bootstrap: int = 200,
        only_selected: bool = True,
        ax=None,
    ):
        """
        Forest plot of debiased coefficient estimates with confidence intervals.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the CIs.
        strategy : str, default "asymptotic"
            Which CI strategy to display. One of ``"asymptotic"`` or
            ``"bootstrap"``.
        n_bootstrap : int, default 200
            Number of bootstrap resamples (only used when
            ``strategy="bootstrap"``).
        only_selected : bool, default True
            If True (default), plot only Lasso-selected features. If False,
            plot all features.
        ax : matplotlib.axes.Axes or None, default None
            Axes to draw on. A new figure is created if None.

        Returns
        -------
        ax : matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If no features are available to plot or strategy is unknown.
        """
        self._check_fitted()

        if strategy == "asymptotic":
            df = self.confidence_intervals(alpha=alpha)
            label = f"Debiased ({int((1 - alpha) * 100)}% CI)"
            color = "steelblue"
        elif strategy == "bootstrap":
            df = self.bootstrap_ci(alpha=alpha, n_bootstrap=n_bootstrap)
            label = f"Bootstrap ({int((1 - alpha) * 100)}% CI)"
            color = "darkorange"
        else:
            raise ValueError(
                f"strategy must be 'asymptotic' or 'bootstrap', got '{strategy}'."
            )

        import matplotlib.pyplot as plt

        if only_selected:
            plot_df = df[df["selected"]].copy()
        else:
            plot_df = df.copy()

        if plot_df.empty:
            raise ValueError("No features to plot.")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, len(plot_df) * 0.5 + 1)))

        features = plot_df["feature"].tolist()
        coefs = plot_df["coef"].values
        ci_lo = plot_df["ci_lower"].values
        ci_hi = plot_df["ci_upper"].values

        y_pos = np.arange(len(features))
        ax.errorbar(
            coefs,
            y_pos,
            xerr=[coefs - ci_lo, ci_hi - coefs],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            linewidth=1.5,
            markersize=6,
        )
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Coefficient (log scale)")
        ax.set_title(
            f"PenalizedGLMInference: {self.family.capitalize()} — {label}"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "This PenalizedGLMInference instance is not fitted yet. "
                "Call fit() before accessing inference methods."
            )

    def _build_dataframe(
        self,
        coef_debiased: np.ndarray,
        se: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        pvalues: np.ndarray,
    ) -> pd.DataFrame:
        """Assemble inference results into a DataFrame."""
        rows = []
        for j, name in enumerate(self.feature_names_):
            rows.append({
                "feature": name,
                "coef_penalized": float(self.coef_penalized_[j]),
                "coef": float(coef_debiased[j]),
                "se": float(se[j]),
                "ci_lower": float(ci_lower[j]),
                "ci_upper": float(ci_upper[j]),
                "pvalue": float(pvalues[j]),
                "selected": bool(self.selected_features_[j]),
            })
        return pd.DataFrame(rows)
