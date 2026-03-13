"""
api.py — sklearn-compatible ANAM wrapper.

The sklearn API (fit/predict/score) is the lingua franca for Python ML
libraries. Supporting it means actuaries can drop ANAM into existing
pipelines, cross-validation grids, and comparison benchmarks without
learning a new interface.

Design decisions:
1. X is always a numpy array (or anything array-like). Polars DataFrames
   are accepted and converted internally.
2. Categorical features are identified by column name or index, not by
   dtype. This is explicit and avoids dtype inference bugs.
3. sample_weight maps to exposure in the distributional loss — not to
   arbitrary observation weights. This is important: doubling exposure
   for a policy is not the same as duplicating the observation.
4. predict() returns mu (expected claims or pure premium per policy year).
5. score() returns negative mean deviance (higher is better, sklearn
   convention for scoring methods).

ANAM is not a pipeline step that transforms features — it's a terminal
estimator. Feature normalisation should happen before calling fit().
The wrapper handles normalisation of continuous features internally if
normalize=True (default).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch

from .model import ANAMModel, FeatureConfig, InteractionConfig
from .shapes import ShapeFunction, extract_shape_functions
from .trainer import ANAMTrainer, TrainingConfig, TrainingHistory
from .utils import StandardScaler, compute_deviance_stat


class ANAM:
    """sklearn-compatible Actuarial Neural Additive Model.

    Parameters
    ----------
    feature_configs:
        List of FeatureConfig objects defining features. If None, all
        features are treated as continuous with no constraints.
    feature_names:
        Feature names (used when feature_configs is None). Must match
        columns of X passed to fit().
    categorical_features:
        List of feature names that are categorical. Only used when
        feature_configs is None.
    monotone_increasing:
        Feature names to constrain as monotone increasing.
    monotone_decreasing:
        Feature names to constrain as monotone decreasing.
    link:
        Link function: 'log' (Poisson/Tweedie), 'identity' (Gaussian),
        'logit' (binary).
    loss:
        Distributional loss: 'poisson', 'tweedie', 'gamma', 'mse'.
    tweedie_p:
        Tweedie power parameter (only for loss='tweedie').
    interaction_pairs:
        List of (feature_i, feature_j) tuples for interaction subnetworks.
    hidden_sizes:
        Default hidden layer sizes for subnetworks.
    n_epochs:
        Maximum training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam learning rate.
    lambda_smooth:
        Smoothness regularisation weight.
    lambda_l2:
        L2 ridge weight.
    lambda_l1:
        L1 sparsity weight.
    patience:
        Early stopping patience (epochs).
    normalize:
        If True, standardise continuous features before training. The
        scaler is stored and applied automatically during predict().
    verbose:
        Training verbosity. 0 = silent.
    device:
        'cpu', 'cuda', or None (auto).
    """

    def __init__(
        self,
        feature_configs: Optional[List[FeatureConfig]] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        monotone_increasing: Optional[List[str]] = None,
        monotone_decreasing: Optional[List[str]] = None,
        link: Literal["log", "identity", "logit"] = "log",
        loss: Literal["poisson", "tweedie", "gamma", "mse"] = "poisson",
        tweedie_p: float = 1.5,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        hidden_sizes: Optional[List[int]] = None,
        n_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        lambda_smooth: float = 1e-4,
        lambda_l2: float = 1e-4,
        lambda_l1: float = 0.0,
        patience: int = 15,
        normalize: bool = True,
        verbose: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.feature_configs = feature_configs
        self.feature_names = feature_names
        # Store exactly as passed — no coercion to [] here.
        # sklearn clone() checks that get_params() round-trips through __init__
        # exactly. Coercing None->[] would break that invariant.
        # Defaults are applied at point-of-use in fit() and _build_feature_configs().
        self.categorical_features = categorical_features
        self.monotone_increasing = monotone_increasing
        self.monotone_decreasing = monotone_decreasing
        self.link = link
        self.loss = loss
        self.tweedie_p = tweedie_p
        self.interaction_pairs = interaction_pairs
        self.hidden_sizes = hidden_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_smooth = lambda_smooth
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.patience = patience
        self.normalize = normalize
        self.verbose = verbose
        self.device = device

        # Set after fit()
        self.model_: Optional[ANAMModel] = None
        self.scaler_: Optional[StandardScaler] = None
        self.history_: Optional[TrainingHistory] = None
        self.feature_names_in_: Optional[List[str]] = None
        self._continuous_col_indices: List[int] = []
        # P0-1 fix: track both the cached shapes and the n_points used to
        # build them so we can detect stale cache on different n_points.
        self._shapes_cache: Optional[Dict[str, ShapeFunction]] = None
        self._shapes_cache_n_points: Optional[int] = None
        self._X_train_scaled: Optional[np.ndarray] = None
        # P0-2 fix: per-feature category remapping tables set during fit().
        # Maps feature name -> array where remap[original_code] = new_code.
        self._cat_remap_: Dict[str, np.ndarray] = {}

    def fit(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        sample_weight: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> "ANAM":
        """Fit the ANAM model.

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p). Polars DataFrame accepted.
        y:
            Target vector (claim counts, rates, or severities).
        sample_weight:
            Exposure weights. For frequency models this is policy duration
            (e.g. years on risk). If None, uniform exposure assumed.

        Returns
        -------
        self
        """
        X_arr, y_arr, w_arr, names = self._validate_input(X, y, sample_weight)
        self.feature_names_in_ = names

        # Build feature configs if not provided. Also populates self._cat_remap_
        # for any categorical columns that are not zero-indexed consecutive ints.
        feat_configs = self.feature_configs or self._build_feature_configs(names, X_arr)

        # P0-2 fix: apply category remapping to training data so that the
        # trainer and scaler see zero-indexed codes, matching the embedding
        # table sizes that _build_feature_configs computed.
        if self._cat_remap_:
            X_arr = self._apply_cat_remap(X_arr)

        # Identify continuous column indices for normalisation
        self._continuous_col_indices = [
            i for i, cfg in enumerate(feat_configs)
            if cfg.feature_type == "continuous"
        ]

        # Normalise continuous features
        if self.normalize and self._continuous_col_indices:
            self.scaler_ = StandardScaler()
            cont_cols = np.array(self._continuous_col_indices)
            X_arr_norm = X_arr.copy()
            X_arr_norm[:, cont_cols] = self.scaler_.fit(
                X_arr[:, cont_cols], feature_names=[names[i] for i in cont_cols]
            ).transform(X_arr[:, cont_cols])
        else:
            X_arr_norm = X_arr

        self._X_train_scaled = X_arr_norm

        # Build interaction configs
        interaction_configs: List[InteractionConfig] = []
        for fi, fj in (self.interaction_pairs or []):
            interaction_configs.append(
                InteractionConfig(feature_i=fi, feature_j=fj)
            )

        # Build model
        self.model_ = ANAMModel(
            feature_configs=feat_configs,
            link=self.link,
            interaction_configs=interaction_configs,
            hidden_sizes=self.hidden_sizes or [64, 32],
        )

        # Build trainer
        train_cfg = TrainingConfig(
            loss=self.loss,
            tweedie_p=self.tweedie_p,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            lambda_smooth=self.lambda_smooth,
            lambda_l2=self.lambda_l2,
            lambda_l1=self.lambda_l1,
            patience=self.patience,
            verbose=self.verbose,
            device=self.device,
        )
        trainer = ANAMTrainer(self.model_, train_cfg)
        self.history_ = trainer.fit(X_arr_norm, y_arr, exposure=w_arr)
        self.model_ = trainer.model  # may have moved to device

        return self

    def predict(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        exposure: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> np.ndarray:
        """Predict expected mean (mu) for each observation.

        Parameters
        ----------
        X:
            Feature matrix.
        exposure:
            Exposure for each row. If None, exposure=1 assumed (predicts
            per-policy-year rate for log-link models).

        Returns
        -------
        np.ndarray
            Predicted means, shape (n,).
        """
        self._check_fitted()

        X_arr = self._to_array(X)
        X_arr = self._apply_cat_remap(X_arr)
        X_arr = self._apply_scaling(X_arr)

        if exposure is not None:
            if isinstance(exposure, pl.Series):
                exposure = exposure.to_numpy()
            exp_arr = np.asarray(exposure, dtype=np.float32)
            log_exp = torch.tensor(np.log(exp_arr.clip(min=1e-8)), dtype=torch.float32)
        else:
            log_exp = None

        X_t = torch.tensor(X_arr, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            mu = self.model_(X_t, log_exposure=log_exp)

        return mu.cpu().numpy()

    def score(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        sample_weight: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> float:
        """Return negative mean deviance (higher = better).

        Follows sklearn convention where score() returns a value where
        higher is better. We negate deviance so that model.score() can
        be maximised by hyperparameter search.
        """
        self._check_fitted()

        y_arr = np.asarray(y if not isinstance(y, pl.Series) else y.to_numpy(), dtype=np.float32)
        y_pred = self.predict(X, exposure=sample_weight)

        w = None
        if sample_weight is not None:
            w = np.asarray(
                sample_weight if not isinstance(sample_weight, pl.Series)
                else sample_weight.to_numpy(),
                dtype=np.float32,
            )

        dev = compute_deviance_stat(
            y_arr, y_pred, exposure=w,
            loss=self.loss, tweedie_p=self.tweedie_p
        )
        return -dev

    def shape_functions(
        self,
        n_points: int = 200,
        category_labels: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Dict[str, ShapeFunction]:
        """Extract and cache shape functions for all features.

        Returns a dict mapping feature name to ShapeFunction. Shape
        functions are evaluated over the observed training data range.

        The cache is keyed on n_points. Calling with a different n_points
        than the cached value discards the stale cache and recomputes.
        """
        self._check_fitted()
        assert self._X_train_scaled is not None

        # P0-1 fix: invalidate cache when n_points changes.
        if self._shapes_cache is None or self._shapes_cache_n_points != n_points:
            self._shapes_cache = extract_shape_functions(
                self.model_,
                self._X_train_scaled,
                n_points=n_points,
                category_labels=category_labels,
            )
            self._shapes_cache_n_points = n_points
        return self._shapes_cache

    def feature_importance(self) -> pl.DataFrame:
        """Feature importance as subnetwork weight norms.

        Returns a Polars DataFrame with columns [feature, importance],
        sorted by importance descending.
        """
        self._check_fitted()
        imp = self.model_.feature_importance()
        return pl.DataFrame(
            {"feature": list(imp.keys()), "importance": list(imp.values())}
        ).sort("importance", descending=True)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """sklearn get_params for grid search compatibility.

        Returns ALL constructor parameters — including structural ones
        (feature_configs, interaction_pairs, etc.) — so that
        sklearn.base.clone() can reconstruct a functionally equivalent
        model.
        """
        # P0-4 fix: include all constructor parameters, not just numerical ones.
        return {
            "feature_configs": self.feature_configs,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "monotone_increasing": self.monotone_increasing,
            "monotone_decreasing": self.monotone_decreasing,
            "link": self.link,
            "loss": self.loss,
            "tweedie_p": self.tweedie_p,
            "interaction_pairs": self.interaction_pairs,
            "hidden_sizes": self.hidden_sizes,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lambda_smooth": self.lambda_smooth,
            "lambda_l2": self.lambda_l2,
            "lambda_l1": self.lambda_l1,
            "patience": self.patience,
            "normalize": self.normalize,
            "verbose": self.verbose,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "ANAM":
        """sklearn set_params for grid search compatibility."""
        for key, val in params.items():
            setattr(self, key, val)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict() or score().")

    def _to_array(self, X: Union[np.ndarray, pl.DataFrame]) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            return X.to_numpy()
        return np.asarray(X, dtype=np.float64)

    def _apply_scaling(self, X_arr: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or not self._continuous_col_indices:
            return X_arr.astype(np.float32)
        X_norm = X_arr.copy().astype(np.float32)
        cont_cols = np.array(self._continuous_col_indices)
        X_norm[:, cont_cols] = self.scaler_.transform(X_arr[:, cont_cols].astype(np.float64)).astype(np.float32)
        return X_norm

    def _apply_cat_remap(self, X_arr: np.ndarray) -> np.ndarray:
        """Apply stored category remappings.

        P0-2 fix: if any categorical features were remapped during fit
        (because they were not zero-indexed consecutive integers), apply
        the same remapping here. Called both during fit() (on training
        data) and during predict() (on new data) so that codes always
        align with the trained embedding table.
        """
        if not self._cat_remap_:
            return X_arr
        if self.feature_names_in_ is None:
            return X_arr
        X_out = X_arr.copy()
        for feat_name, remap in self._cat_remap_.items():
            if feat_name in self.feature_names_in_:
                col_idx = self.feature_names_in_.index(feat_name)
                codes = X_out[:, col_idx].astype(int)
                X_out[:, col_idx] = remap[codes].astype(X_out.dtype)
        return X_out

    def _validate_input(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Union[np.ndarray, pl.Series],
        w: Optional[Union[np.ndarray, pl.Series]],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
        """Convert inputs to numpy and extract feature names."""
        if isinstance(X, pl.DataFrame):
            names = X.columns
            X_arr = X.to_numpy().astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            names = self.feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

        if isinstance(y, pl.Series):
            y_arr = y.to_numpy().astype(np.float32)
        else:
            y_arr = np.asarray(y, dtype=np.float32)

        if w is not None:
            if isinstance(w, pl.Series):
                w_arr: Optional[np.ndarray] = w.to_numpy().astype(np.float32)
            else:
                w_arr = np.asarray(w, dtype=np.float32)
        else:
            w_arr = None

        return X_arr, y_arr, w_arr, list(names)

    def _build_feature_configs(
        self, names: List[str], X_arr: np.ndarray
    ) -> List[FeatureConfig]:
        """Auto-construct FeatureConfig list from names and constraints.

        P0-2 fix: for categorical features, validate that column values are
        zero-indexed consecutive integers. If not, auto-remap and warn. The
        remap is stored in self._cat_remap_ and applied to the training
        array in fit() before the scaler and trainer receive it.
        """
        configs: List[FeatureConfig] = []
        self._cat_remap_ = {}

        for i, name in enumerate(names):
            if name in (self.categorical_features or []):
                col = X_arr[:, i].astype(int)
                unique_vals = np.unique(col)
                expected = np.arange(len(unique_vals))
                if not np.array_equal(unique_vals, expected):
                    # Not zero-indexed consecutive integers. Auto-remap.
                    warnings.warn(
                        f"Categorical feature '{name}' has values {unique_vals.tolist()} "
                        f"which are not zero-indexed consecutive integers. "
                        f"Auto-remapping to 0..{len(unique_vals) - 1}. "
                        f"The same remapping will be applied at predict time. "
                        f"If you rely on specific category indices in post-hoc analysis, "
                        f"remap the column yourself before passing to fit().",
                        UserWarning,
                        stacklevel=3,
                    )
                    # Build a lookup array: remap[original_code] = new_code.
                    max_orig = int(unique_vals.max())
                    remap = np.full(max_orig + 1, -1, dtype=int)
                    for new_code, orig_code in enumerate(unique_vals):
                        remap[orig_code] = new_code
                    self._cat_remap_[name] = remap
                    col = remap[col]

                n_cats = int(col.max()) + 1
                configs.append(
                    FeatureConfig(
                        name=name,
                        feature_type="categorical",
                        n_categories=n_cats,
                    )
                )
            else:
                mono: Literal["increasing", "decreasing", "none"] = "none"
                if name in (self.monotone_increasing or []):
                    mono = "increasing"
                elif name in (self.monotone_decreasing or []):
                    mono = "decreasing"
                configs.append(
                    FeatureConfig(
                        name=name,
                        feature_type="continuous",
                        monotonicity=mono,
                    )
                )
        return configs
