"""
utils.py — Integration utilities: interaction selection, GLM comparison,
           data preprocessing helpers.

Interaction selection:
The simplest principled approach for insurance is correlation-based
screening: compute pairwise Pearson correlation between features, then
flag pairs with |r| > threshold as candidates for interaction terms.
This is conservative but defensible. The FAST algorithm (from EBM) is
more powerful but requires significantly more implementation. For most
insurance pricing tasks, correlation screening finds the important pairs.

GLM comparison utilities:
Actuaries reviewing ANAM outputs want to compare against an existing GLM.
These utilities convert shape functions to GLM-style log-relativities and
compute a deviation summary table.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------


class StandardScaler:
    """Simple StandardScaler that tracks feature names and ranges.

    Stores mean/std for inverse transformation (needed to recover original
    feature values when plotting shape functions).
    """

    def __init__(self) -> None:
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "StandardScaler":
        self.means_ = X.mean(axis=0)
        self.stds_ = X.std(axis=0).clip(min=1e-8)
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.means_) / self.stds_

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return X_scaled * self.stds_ + self.means_

    def inverse_transform_col(self, x_scaled: np.ndarray, col_idx: int) -> np.ndarray:
        """Inverse transform a single column."""
        assert self.means_ is not None
        return x_scaled * self.stds_[col_idx] + self.means_[col_idx]


# ---------------------------------------------------------------------------
# Interaction selection
# ---------------------------------------------------------------------------


def select_interactions_correlation(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.3,
    top_k: Optional[int] = 10,
    exclude_categorical: Optional[List[int]] = None,
) -> List[Tuple[str, str, float]]:
    """Screen feature pairs for interaction candidates using Pearson correlation.

    Pairs with |r| above threshold are candidates for interaction subnetworks.
    Lower |r| means the two features are more independent in the feature space
    — which paradoxically can mean their interaction is more surprising and
    worth modelling. This heuristic selects the *highest* correlation pairs
    because they often represent rate structures where the joint effect differs
    from the sum of main effects.

    Parameters
    ----------
    X:
        Feature matrix (n, p). Should be the continuous features only.
    feature_names:
        Names matching columns of X.
    threshold:
        Minimum |correlation| to include pair. Default 0.3.
    top_k:
        Maximum number of pairs to return. None = all above threshold.
    exclude_categorical:
        Column indices of categorical features to exclude from screening.

    Returns
    -------
    List[Tuple[str, str, float]]
        List of (feature_i, feature_j, correlation) sorted by |r| descending.
    """
    exclude = set(exclude_categorical or [])
    p = X.shape[1]

    candidates: List[Tuple[str, str, float]] = []

    for i in range(p):
        if i in exclude:
            continue
        for j in range(i + 1, p):
            if j in exclude:
                continue
            corr = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
            if abs(corr) >= threshold:
                candidates.append((feature_names[i], feature_names[j], corr))

    candidates.sort(key=lambda x: abs(x[2]), reverse=True)

    if top_k is not None:
        candidates = candidates[:top_k]

    return candidates


def select_interactions_residual(
    X: np.ndarray,
    y_residuals: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
    exclude_categorical: Optional[List[int]] = None,
) -> List[Tuple[str, str, float]]:
    """Select interaction pairs by pairwise product correlation with residuals.

    After fitting the additive model, residuals contain unexplained variance.
    This method checks whether x_i * x_j correlates with residuals — if so,
    the interaction term x_i x_j may be worth adding.

    More principled than pure feature-space correlation for identifying
    interactions that actually improve model fit.

    Parameters
    ----------
    X:
        Feature matrix.
    y_residuals:
        Model residuals (y - predicted).
    feature_names:
        Feature names.
    top_k:
        Number of top pairs to return.

    Returns
    -------
    List[Tuple[str, str, float]]
        Sorted by |correlation with residuals| descending.
    """
    exclude = set(exclude_categorical or [])
    p = X.shape[1]
    candidates: List[Tuple[str, str, float]] = []

    for i in range(p):
        if i in exclude:
            continue
        for j in range(i + 1, p):
            if j in exclude:
                continue
            product = X[:, i] * X[:, j]
            corr = float(np.corrcoef(product, y_residuals)[0, 1])
            candidates.append((feature_names[i], feature_names[j], abs(corr)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# GLM comparison utilities
# ---------------------------------------------------------------------------


def shapes_to_relativity_table(
    shapes: Dict[str, "ShapeFunction"],
    feature_names: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Aggregate all shape functions into a single relativity table.

    Outputs a long-format DataFrame with one row per (feature, level)
    combination, suitable for actuarial review in Excel.

    Parameters
    ----------
    shapes:
        Dict returned by extract_shape_functions().
    feature_names:
        Subset of feature names to include. None = all features.

    Returns
    -------
    pl.DataFrame
        Columns: [feature, level, f_x, relativity, log_relativity]
    """
    names = feature_names or list(shapes.keys())
    dfs: List[pl.DataFrame] = []

    for name in names:
        if name not in shapes:
            continue
        sf = shapes[name]
        rel_df = sf.to_relativities()

        if sf.feature_type == "continuous":
            level_col = pl.Series("level", [str(round(float(x), 4)) for x in sf.x_values])
        else:
            level_col = pl.Series(
                "level",
                [
                    sf.category_labels.get(int(i), str(int(i))) if sf.category_labels else str(int(i))
                    for i in sf.x_values
                ],
            )

        feature_col = pl.Series("feature", [name] * len(sf.x_values))
        f_x_col = pl.Series("f_x", sf.f_values.tolist())

        rel_col = rel_df["relativity"]
        log_rel_col = rel_df["log_relativity"]

        dfs.append(
            pl.DataFrame(
                {
                    "feature": feature_col,
                    "level": level_col,
                    "f_x": f_x_col,
                    "relativity": rel_col,
                    "log_relativity": log_rel_col,
                }
            )
        )

    if not dfs:
        return pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "level": pl.Utf8,
                "f_x": pl.Float64,
                "relativity": pl.Float64,
                "log_relativity": pl.Float64,
            }
        )

    return pl.concat(dfs)


def compare_shapes_to_glm(
    anam_shapes: Dict[str, "ShapeFunction"],
    glm_coefficients: Dict[str, Dict[str, float]],
) -> pl.DataFrame:
    """Compare ANAM shape functions to GLM log-relativities.

    For each feature level in the GLM, find the nearest ANAM shape function
    value and compute the deviation. Useful for model validation against
    existing production GLMs.

    Parameters
    ----------
    anam_shapes:
        ANAM shape functions (from extract_shape_functions).
    glm_coefficients:
        Dict mapping feature_name -> {level_str -> log_relativity}.
        GLM log-relativities in the same scale as ANAM f_i outputs.

    Returns
    -------
    pl.DataFrame
        Columns: [feature, level, anam_f, glm_log_rel, deviation]
    """
    rows: List[Dict] = []

    for feature_name, glm_levels in glm_coefficients.items():
        if feature_name not in anam_shapes:
            continue

        sf = anam_shapes[feature_name]

        for level_str, glm_val in glm_levels.items():
            # Find ANAM value at this level
            if sf.feature_type == "continuous":
                try:
                    x_level = float(level_str)
                    idx = int(np.argmin(np.abs(sf.x_values - x_level)))
                    anam_val = float(sf.f_values[idx])
                except (ValueError, IndexError):
                    continue
            else:
                try:
                    cat_idx = int(level_str)
                    if cat_idx < len(sf.f_values):
                        anam_val = float(sf.f_values[cat_idx])
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

            rows.append(
                {
                    "feature": feature_name,
                    "level": level_str,
                    "anam_f": anam_val,
                    "glm_log_rel": glm_val,
                    "deviation": anam_val - glm_val,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "level": pl.Utf8,
                "anam_f": pl.Float64,
                "glm_log_rel": pl.Float64,
                "deviation": pl.Float64,
            }
        )

    return pl.DataFrame(rows)


def compute_deviance_stat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exposure: Optional[np.ndarray] = None,
    loss: str = "poisson",
    tweedie_p: float = 1.5,
    eps: float = 1e-8,
) -> float:
    """Compute weighted deviance statistic for model comparison.

    Returns the mean deviance (lower is better). Useful for comparing
    ANAM to GLM/EBM baselines with a single number.
    """
    import torch

    y_t = torch.tensor(y_true, dtype=torch.float32)
    yp_t = torch.tensor(y_pred, dtype=torch.float32)
    w_t = torch.tensor(exposure, dtype=torch.float32) if exposure is not None else None

    from .losses import gamma_deviance, poisson_deviance, tweedie_deviance

    if loss == "poisson":
        return float(poisson_deviance(yp_t, y_t, w_t, eps).item())
    elif loss == "tweedie":
        return float(tweedie_deviance(yp_t, y_t, tweedie_p, w_t, eps).item())
    elif loss == "gamma":
        return float(gamma_deviance(yp_t, y_t, w_t, eps).item())
    else:
        raise ValueError(f"Unknown loss: {loss!r}")
