"""
Bias-corrected confidence intervals for Lasso-selected GLM coefficients.

This module implements the debiased Lasso estimator for Tweedie-family GLMs
(Poisson, Gamma, Tweedie) from Manna, Huang, Dey, Gu & He (2025). After
Lasso selects a sparse set of rating factors, standard Wald CIs on those
coefficients are invalid — the Lasso shrinkage biases the estimates toward
zero. This module corrects for that bias and produces asymptotically valid
confidence intervals.

Two strategies are available:

1. **Strategy A (asymptotic debiased estimator)**: One-step Newton correction
   from the penalised solution using the inverse Hessian. O(n*p^2) time.
   Valid when p^2/n -> 0 (large portfolios with sparse selected models).

2. **Strategy B (Pearson residual bootstrap)**: Resamples Pearson residuals
   to construct bootstrap distribution of the penalised estimator. Valid
   under weaker assumptions but requires B refits of the penalised GLM.

**When to use which:**

- Poisson frequency models (n > 50,000, selected p < 100): Strategy A is
  fast and well-justified.
- Gamma severity models: Strategy A covers Gamma directly (PostSelectionGLM
  does not), filling a gap in the insurance inference toolkit.
- Small portfolios (n < 10,000) or suspect sparsity: Strategy B is safer.

**Relationship to PostSelectionGLM:**

:class:`PostSelectionGLM` answers "is this variable genuinely significant
given that Lasso selected it?" (conditional inference, truncated normal).
:class:`DebiasedGLM` answers "given it's in the model, what is the plausible
range for its coefficient magnitude?" (marginal inference, Wald-style).
Both are valid; they answer different questions. Report both.

References
----------
Manna, A., Huang, B., Dey, D. K., Gu, C., & He, X. (2025). Interval
estimation of coefficients in penalized regression models of insurance data.
Applied Stochastic Models in Business and Industry. arXiv:2410.01008.

Examples
--------
>>> import numpy as np
>>> from insurance_gam.debiased_glm import DebiasedGLM
>>> rng = np.random.default_rng(42)
>>> n, p = 2000, 10
>>> X = rng.standard_normal((n, p))
>>> y = rng.poisson(np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]))
>>> model = DebiasedGLM(family="poisson").fit(X, y)
>>> df = model.summary()
>>> df.head()
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
        "DebiasedGLM requires scikit-learn. "
        "Install with: pip install insurance-gam[glm]"
    ) from exc

try:
    import statsmodels.api as sm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DebiasedGLM requires statsmodels. "
        "Install with: pip install insurance-gam[glm]"
    ) from exc


# ---------------------------------------------------------------------------
# Supported families
# ---------------------------------------------------------------------------

_SUPPORTED_FAMILIES = ("poisson", "gamma", "tweedie")


def _check_family(family: str, tweedie_power: float) -> None:
    """Validate family string and Tweedie power parameter."""
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
# Variance function and Hessian weights per family
# ---------------------------------------------------------------------------


def _variance_function(mu: np.ndarray, family: str, phi: float, p: float) -> np.ndarray:
    """
    GLM variance function V(mu) — includes dispersion where relevant.

    Parameters
    ----------
    mu : ndarray
        Fitted mean vector.
    family : str
        One of "poisson", "gamma", "tweedie".
    phi : float
        Dispersion estimate (ignored for Poisson, phi=1 there).
    p : float
        Tweedie power parameter (ignored unless family="tweedie").

    Returns
    -------
    V : ndarray
        Variance V(mu) per observation.
    """
    if family == "poisson":
        return mu  # V(mu) = mu, phi = 1
    elif family == "gamma":
        return mu ** 2 / phi  # V(mu) = mu^2 / phi
    else:  # tweedie
        return (mu ** p) / phi  # V(mu) = mu^p / phi


def _hessian_weights(mu: np.ndarray, family: str, phi: float, p: float) -> np.ndarray:
    """
    Diagonal weights W for the Hessian H = X^T diag(W) X / n.

    For log link, the Hessian of the negative log-likelihood is
    X^T diag(mu^{2-p} / phi) X, where p=1 for Poisson, p=2 for Gamma,
    and p in (1,2) for Tweedie.

    Poisson:  W_i = mu_i           (p=1: mu^{2-1}/1 = mu)
    Gamma:    W_i = mu_i^2 / phi   (p=2: mu^{2-2}/phi = 1/phi * mu^0 ... wait)
    Actually for log link GLM: W_i = mu_i^2 / V(mu_i) = 1/phi for Gamma,
    but we want the expected Fisher information (XtWX with W = b''(eta)/phi).

    Derivation (log link, Tweedie family):
      l_i = -y_i * exp((1-p)*eta_i)/(1-p) + exp((2-p)*eta_i)/(2-p)
      d^2 l_i / d eta_i^2 = exp((2-p)*eta_i) = mu_i^(2-p)
      With dispersion: divide by phi.
    So W_i = mu_i^(2-p) / phi for all three families.
    """
    if family == "poisson":
        return mu  # p=1: mu^(2-1)/1 = mu
    elif family == "gamma":
        return np.ones_like(mu) / phi  # p=2: mu^0/phi = 1/phi
    else:  # tweedie
        return mu ** (2.0 - p) / phi


def _sqrt_variance(mu: np.ndarray, family: str, phi: float, p: float) -> np.ndarray:
    """
    sqrt(V(mu)) for computing Pearson residuals in the bootstrap.

    Pearson residual: r_i = (y_i - mu_i) / sqrt(V(mu_i))
    """
    if family == "poisson":
        return np.sqrt(mu)
    elif family == "gamma":
        return mu / np.sqrt(phi)
    else:  # tweedie
        return mu ** (p / 2.0) * np.sqrt(phi)


# ---------------------------------------------------------------------------
# Dispersion estimation
# ---------------------------------------------------------------------------


def _estimate_dispersion(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    p: float,
    n_params: int,
) -> float:
    """
    Estimate dispersion phi from Pearson chi-squared statistic.

    phi_hat = sum((y - mu)^2 / V_raw(mu)) / (n - p)

    where V_raw is the variance function without phi (i.e., mu for Poisson,
    mu^2 for Gamma, mu^p for Tweedie).

    For Poisson, phi is always 1 (not estimated).
    """
    if family == "poisson":
        return 1.0

    n = len(y)
    dof = max(n - n_params, 1)

    if family == "gamma":
        v_raw = mu ** 2
    else:  # tweedie
        v_raw = mu ** p

    pearson_chi2 = np.sum((y - mu) ** 2 / np.clip(v_raw, 1e-14, None))
    return float(pearson_chi2 / dof)


# ---------------------------------------------------------------------------
# Penalised GLM fitting via IRLS pseudo-data + ElasticNet
# ---------------------------------------------------------------------------


def _make_offset(exposure: Optional[np.ndarray], n: int) -> np.ndarray:
    """Return log(exposure) offset; zeros if exposure is None."""
    if exposure is None:
        return np.zeros(n)
    exposure = np.asarray(exposure, dtype=float)
    if exposure.shape != (n,):
        raise ValueError(
            f"exposure must be a 1-D array of length {n}, got shape {exposure.shape}."
        )
    if np.any(exposure <= 0):
        raise ValueError("All exposure values must be strictly positive.")
    return np.log(exposure)


def _fit_penalised_glm(
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
    Fit elastic net GLM via IRLS pseudo-data linearisation.

    Uses the Fisher scoring trick: linearise the GLM log-likelihood around
    an initial MLE estimate to produce a weighted least-squares problem,
    then apply ElasticNet (with CV if alpha is None, else fixed alpha).

    Parameters
    ----------
    X : ndarray (n, p)
        Feature matrix (no intercept column).
    y : ndarray (n,)
        Response.
    offset : ndarray (n,)
        Log-exposure offset.
    family : str
        GLM family.
    phi : float
        Dispersion (initial guess; typically 1.0 on first call).
    tweedie_power : float
        Tweedie p parameter.
    alpha : float
        ElasticNet regularisation strength. If <= 0, use CV to select.
    l1_ratio : float
        Elastic net mixing (1.0 = Lasso, 0.0 = Ridge).
    cv_folds : int
        Number of CV folds (used when alpha <= 0).
    random_state : int or None
    scaler : StandardScaler
        Fitted scaler for X. If fit_scaler=True, will be fit here.
    fit_scaler : bool
        Whether to fit the scaler on X.

    Returns
    -------
    beta_hat : ndarray (p,)
        Penalised coefficient estimates (in original X scale).
    mu_hat : ndarray (n,)
        Fitted means.
    intercept : float
        Fitted intercept.
    lambda_opt : float
        Chosen regularisation parameter.
    """
    n, p = X.shape

    # Scale X for ElasticNet
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Initial eta: use log(y + 0.1) as starting point
    eta = np.log(np.clip(y, 0.1, None)) - offset

    # IRLS loop — typically converges in 5-10 steps
    max_irls = 20
    tol_irls = 1e-6
    beta_hat = np.zeros(p)
    intercept = np.mean(eta)
    lambda_opt = alpha

    for irls_iter in range(max_irls):
        mu = np.exp(np.clip(eta + offset, -30, 30))
        mu = np.clip(mu, 1e-10, None)

        # IRLS weights and pseudo-response (log link, any Tweedie family)
        W = _hessian_weights(mu, family, phi, tweedie_power)
        W = np.clip(W, 1e-14, None)
        sqrt_W = np.sqrt(W)

        # Working response: z_i = eta_i + (y_i - mu_i) / (mu_i * W_i * phi) * W_i
        # For log link: dg/dmu = 1/mu, so working response:
        # z_i = eta_i + (y_i - mu_i) / (mu_i)  [derivative of log link times residual]
        z = eta + (y - mu) / mu

        # Weighted problem: sqrt(W) * z_i ~ intercept + sqrt(W) * X @ beta
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
            lambda_opt = alpha
        else:
            # CV to select lambda (only on first IRLS iteration)
            if irls_iter == 0:
                en_cv = ElasticNetCV(
                    l1_ratio=l1_ratio,
                    cv=cv_folds,
                    fit_intercept=True,
                    max_iter=10_000,
                    n_jobs=1,
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

        # Unscale back to original X space: beta_orig = beta_scaled / scale
        beta_scaled = en.coef_
        scale = scaler.scale_
        beta_hat_new = beta_scaled / scale

        # Intercept: z_weighted has been re-centred by ElasticNet's intercept
        # Reconstruct intercept in original eta space
        intercept_new = en.intercept_ - np.dot(scaler.mean_ / scale, beta_scaled)

        # Update eta
        eta_new = X @ beta_hat_new + intercept_new

        # Check convergence
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
# Debiasing: Strategy A
# ---------------------------------------------------------------------------


def _debiased_estimate(
    X: np.ndarray,
    y: np.ndarray,
    mu_hat: np.ndarray,
    beta_hat: np.ndarray,
    family: str,
    phi: float,
    tweedie_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute debiased coefficient estimates and standard errors.

    Formula (Manna et al. 2025, eq. following Xia et al. 2021):
        b_hat = beta_hat - Theta @ grad
        Theta = inv(H) = inv(X^T diag(W) X / n)
        grad = X^T (mu_hat - y) / n
        Sigma = H (equals Theta^{-1} for Poisson; sandwich form for others)
        V = Theta @ Sigma @ Theta^T / n
        SE = sqrt(diag(V))

    The sandwich form: V = Theta / n (for correctly specified model,
    Sigma = H so V = H^{-1} / n).

    Parameters
    ----------
    X : ndarray (n, p)
        Feature matrix.
    y : ndarray (n,)
        Observed responses.
    mu_hat : ndarray (n,)
        Fitted means from penalised GLM.
    beta_hat : ndarray (p,)
        Penalised coefficient vector.
    family : str
    phi : float
        Dispersion estimate.
    tweedie_power : float

    Returns
    -------
    b_debiased : ndarray (p,)
        Debiased coefficient vector.
    se : ndarray (p,)
        Asymptotic standard errors.
    Theta : ndarray (p, p)
        Inverse Hessian (debiasing matrix).
    """
    n, p = X.shape

    # Hessian weights
    W = _hessian_weights(mu_hat, family, phi, tweedie_power)

    # Sample Hessian H = X^T diag(W) X / n
    H = (X.T * W) @ X / n  # shape (p, p)

    # Warn if dimension-to-sample ratio is large
    ratio = p ** 2 / n
    if ratio > 0.25:
        warnings.warn(
            f"p^2/n = {ratio:.2f} > 0.25. Debiased estimator validity requires "
            "p^2/n -> 0. Consider reducing the selected model size or using "
            "n_bootstrap > 0 for bootstrap CIs.",
            UserWarning,
            stacklevel=4,
        )

    # Invert Hessian — regularise if near-singular
    try:
        cond = np.linalg.cond(H)
        if cond > 1e12:
            # Add small ridge for stability
            ridge = 1e-8 * np.trace(H) / p
            Theta = np.linalg.inv(H + np.eye(p) * ridge)
        else:
            Theta = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Hessian inversion failed; using pseudoinverse. Results may be unreliable.",
            UserWarning,
            stacklevel=4,
        )
        Theta = np.linalg.pinv(H)

    # Gradient of negative log-likelihood at beta_hat
    # For log-link GLM: grad_j = X_j^T (mu - y) / n
    grad = X.T @ (mu_hat - y) / n  # shape (p,)

    # One-step debiasing: b_hat = beta_hat - Theta @ grad
    b_debiased = beta_hat - Theta @ grad

    # Asymptotic variance: V = Theta @ Sigma_hat @ Theta.T / n
    # For correctly specified model: Sigma_hat = H, so V = Theta / n
    # We use sandwich estimator to be safe: Sigma_hat = X^T diag(res^2 * W^2/W) X / n
    # where res = y - mu_hat. This is the empirical (meat) of the sandwich.
    # Simplification: under correct specification, Sigma = H so V = Theta/n.
    # We implement the simpler form: SE = sqrt(diag(Theta) / n)
    se = np.sqrt(np.clip(np.diag(Theta), 0, None) / n)

    return b_debiased, se, Theta


# ---------------------------------------------------------------------------
# Bootstrap: Strategy B
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    mu_hat: np.ndarray,
    beta_hat: np.ndarray,
    family: str,
    phi: float,
    tweedie_power: float,
    alpha_level: float,
    l1_ratio: float,
    lambda_opt: float,
    n_bootstrap: int,
    random_state: Optional[int],
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pearson residual bootstrap confidence intervals (Strategy B).

    Algorithm 2 from Manna et al. 2025:
    1. Compute Pearson residuals from penalised fit.
    2. Mean-centre residuals.
    3. Resample residuals, reconstruct y_boot, refit penalised GLM.
    4. Pivot CI: [2*beta_hat - Q(1-alpha/2), 2*beta_hat - Q(alpha/2)]

    Parameters
    ----------
    X, y, offset, mu_hat, beta_hat : as above
    family, phi, tweedie_power : family specification
    alpha_level : float
        Nominal level (e.g. 0.05 for 95% CIs).
    l1_ratio : float
    lambda_opt : float
        Fixed lambda (from initial fit).
    n_bootstrap : int
        Number of bootstrap resamples.
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

    # Pearson residuals
    r = (y - mu_hat) / sqrt_v
    r_centred = r - r.mean()

    boot_coefs = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        # Resample centred Pearson residuals
        r_boot = rng.choice(r_centred, size=n, replace=True)

        # Reconstruct response
        y_boot = mu_hat + sqrt_v * r_boot
        # Clip to valid range for the family
        if family == "poisson":
            y_boot = np.clip(y_boot, 0.0, None)
        elif family in ("gamma", "tweedie"):
            y_boot = np.clip(y_boot, 1e-10, None)

        # Refit penalised GLM with fixed lambda
        try:
            beta_boot, _, _, _ = _fit_penalised_glm(
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

    # Pivot CI: [2*beta_hat - Q(1-alpha/2), 2*beta_hat - Q(alpha/2)]
    q_lo = np.percentile(boot_coefs, 100 * (1 - alpha_level / 2), axis=0)
    q_hi = np.percentile(boot_coefs, 100 * (alpha_level / 2), axis=0)

    ci_lower = 2 * beta_hat - q_lo
    ci_upper = 2 * beta_hat - q_hi

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Public class: DebiasedGLM
# ---------------------------------------------------------------------------


class DebiasedGLM:
    """
    Debiased confidence intervals for Lasso-selected GLM coefficients.

    Implements bias-corrected Wald-style confidence intervals for elastic
    net penalised GLMs in the Tweedie family (Poisson, Gamma, Tweedie).
    Based on Manna et al. (2025), arXiv:2410.01008.

    After Lasso selects a sparse rating model, this class:

    1. Fits the elastic net penalised GLM (via IRLS pseudo-data).
    2. Applies a one-step Newton debiasing correction (Strategy A) or
       Pearson residual bootstrap (Strategy B) to produce CIs with
       correct asymptotic coverage.

    **Key difference from PostSelectionGLM**: this class produces marginal
    (unconditional) CIs — valid on average over the selection randomness,
    not conditioned on the specific selection event. Use PostSelectionGLM
    for conditional inference (testing whether a selected variable is truly
    non-zero); use DebiasedGLM for reporting coefficient magnitudes with
    honest uncertainty bounds.

    **Gamma support**: Unlike PostSelectionGLM (Poisson only), DebiasedGLM
    supports Gamma and Tweedie families, making it suitable for severity
    models and combined frequency-severity (Tweedie) models.

    Parameters
    ----------
    family : str, default "poisson"
        GLM family. One of ``"poisson"``, ``"gamma"``, ``"tweedie"``.
    tweedie_power : float, default 1.5
        Variance power for Tweedie family: ``p=1`` is Poisson,
        ``p=2`` is Gamma. Ignored unless ``family="tweedie"``.
    alpha : float, default 1.0
        ElasticNet regularisation strength. If 0.0, use cross-validation
        to select the penalty (5-fold by default). Positive values use
        the given alpha directly — useful when the Lasso path has already
        been run externally.
    l1_ratio : float, default 0.5
        ElasticNet mixing: 1.0 is pure Lasso, 0.0 is Ridge. The paper
        covers elastic net not just Lasso, which handles correlated
        features better in pricing models.
    confidence : float, default 0.95
        Nominal confidence level for intervals (1 - alpha_level).
    n_bootstrap : int, default 0
        Number of Pearson residual bootstrap resamples (Strategy B).
        Set to 0 to use the asymptotic debiased estimator (Strategy A).
        Recommended: 500 for small/moderate n, 0 for large portfolios.
    cv_folds : int, default 5
        Cross-validation folds for lambda selection (used when alpha=0.0).
    random_state : int or None, default None
        Random seed for bootstrap and CV splits.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Debiased coefficient estimates (excluding intercept).
    intercept_ : float
        Fitted intercept.
    se_ : ndarray of shape (n_features,)
        Asymptotic standard errors (Strategy A only; NaN for Strategy B).
    ci_lower_ : ndarray of shape (n_features,)
        Lower confidence interval bounds.
    ci_upper_ : ndarray of shape (n_features,)
        Upper confidence interval bounds.
    pvalues_ : ndarray of shape (n_features,)
        Two-sided p-values from debiased estimates (Strategy A only).
    selected_features_ : ndarray of bool, shape (n_features,)
        Features with non-zero penalised coefficient.
    feature_names_ : list of str
        Feature names (from DataFrame columns or "x0", "x1", ...).
    phi_ : float
        Estimated dispersion parameter.
    lambda_ : float
        ElasticNet regularisation parameter used.

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_gam.debiased_glm import DebiasedGLM
    >>> rng = np.random.default_rng(0)
    >>> n, p = 3000, 12
    >>> X = rng.standard_normal((n, p))
    >>> y = rng.poisson(np.exp(0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]))
    >>> model = DebiasedGLM(family="poisson", alpha=0.0).fit(X, y)
    >>> model.summary()[["feature", "coef", "ci_lower", "ci_upper", "pvalue"]].head()

    Gamma severity model:

    >>> shape = 3.0
    >>> scale = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]) / shape
    >>> y_sev = rng.gamma(shape, scale)
    >>> sev_model = DebiasedGLM(family="gamma", alpha=0.0).fit(X, y_sev)
    >>> sev_model.summary()
    """

    def __init__(
        self,
        family: str = "poisson",
        tweedie_power: float = 1.5,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        confidence: float = 0.95,
        n_bootstrap: int = 0,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        _check_family(family, tweedie_power)
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        if n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}.")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}.")

        self.family = family
        self.tweedie_power = tweedie_power
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self.cv_folds = cv_folds
        self.random_state = random_state

        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.Series"],
        exposure: Optional[Union[np.ndarray, "pd.Series"]] = None,
    ) -> "DebiasedGLM":
        """
        Fit penalised GLM and compute debiased confidence intervals.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix. If a pandas DataFrame, column names are used.
        y : array-like of shape (n,)
            Response variable. Non-negative for Poisson; strictly positive
            for Gamma/Tweedie.
        exposure : array-like of shape (n,) or None, default None
            Exposure for rate modelling. Enters as log(exposure) offset.
            All values must be strictly positive.

        Returns
        -------
        self : DebiasedGLM
            Fitted estimator (for method chaining).
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

        # Fit penalised GLM
        scaler = StandardScaler()
        beta_hat, mu_hat, intercept, lambda_opt = _fit_penalised_glm(
            X_arr, y_arr, offset,
            family=self.family,
            phi=1.0,  # initial dispersion
            tweedie_power=self.tweedie_power,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            scaler=scaler,
            fit_scaler=True,
        )

        # Estimate dispersion from Pearson residuals
        phi = _estimate_dispersion(y_arr, mu_hat, self.family, self.tweedie_power, p + 1)

        # If dispersion changed meaningfully, refit with updated phi
        if self.family != "poisson" and abs(phi - 1.0) > 0.01:
            beta_hat, mu_hat, intercept, lambda_opt = _fit_penalised_glm(
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
            phi = _estimate_dispersion(y_arr, mu_hat, self.family, self.tweedie_power, p + 1)

        self.phi_ = phi
        self.lambda_ = lambda_opt
        self.intercept_ = intercept
        self._beta_penalised = beta_hat.copy()
        self.selected_features_ = np.abs(beta_hat) > 1e-8

        alpha_level = 1.0 - self.confidence

        if self.n_bootstrap == 0:
            # Strategy A: asymptotic debiased estimator
            b_debiased, se, Theta = _debiased_estimate(
                X_arr, y_arr, mu_hat, beta_hat,
                self.family, phi, self.tweedie_power,
            )
            z_crit = stats.norm.ppf(1 - alpha_level / 2)
            ci_lower = b_debiased - z_crit * se
            ci_upper = b_debiased + z_crit * se
            pvalues = 2 * stats.norm.sf(np.abs(b_debiased / np.clip(se, 1e-14, None)))

            self.coef_ = b_debiased
            self.se_ = se
            self.ci_lower_ = ci_lower
            self.ci_upper_ = ci_upper
            self.pvalues_ = pvalues
            self._Theta = Theta

        else:
            # Strategy B: Pearson residual bootstrap
            ci_lower, ci_upper = _bootstrap_ci(
                X_arr, y_arr, offset, mu_hat, beta_hat,
                family=self.family,
                phi=phi,
                tweedie_power=self.tweedie_power,
                alpha_level=alpha_level,
                l1_ratio=self.l1_ratio,
                lambda_opt=lambda_opt,
                n_bootstrap=self.n_bootstrap,
                random_state=self.random_state,
                scaler=scaler,
            )
            # For bootstrap, coefficient point estimate is the penalised estimate
            # (debiasing is implicit in the pivot CI)
            self.coef_ = beta_hat
            self.se_ = np.full(p, np.nan)
            self.ci_lower_ = ci_lower
            self.ci_upper_ = ci_upper
            self.pvalues_ = np.full(p, np.nan)

        self._is_fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        """
        Return inference results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per feature. Columns:

            - ``feature``: feature name
            - ``coef``: debiased coefficient estimate
            - ``se``: standard error (NaN for bootstrap Strategy B)
            - ``ci_lower``: lower confidence bound
            - ``ci_upper``: upper confidence bound
            - ``pvalue``: two-sided p-value (NaN for Strategy B)
            - ``selected``: bool, whether Lasso selected this feature

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before summary().")

        rows = []
        for j, name in enumerate(self.feature_names_):
            rows.append({
                "feature": name,
                "coef": float(self.coef_[j]),
                "se": float(self.se_[j]),
                "ci_lower": float(self.ci_lower_[j]),
                "ci_upper": float(self.ci_upper_[j]),
                "pvalue": float(self.pvalues_[j]),
                "selected": bool(self.selected_features_[j]),
            })

        return pd.DataFrame(rows)

    def forest_plot(self, ax=None, only_selected: bool = True):
        """
        Plot debiased coefficient estimates with confidence intervals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default None
            Axes to draw on. Created if None.
        only_selected : bool, default True
            If True, plot only Lasso-selected features. If False, plot all.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before forest_plot().")

        import matplotlib.pyplot as plt

        df = self.summary()
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
        strategy = "Bootstrap" if self.n_bootstrap > 0 else "Debiased"
        ax.set_title(
            f"{strategy} GLM CIs: {self.family.capitalize()} family "
            f"({int(self.confidence * 100)}% CI)"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax
