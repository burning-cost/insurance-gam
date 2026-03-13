"""
shapes.py — Shape function extraction, plotting, and export.

The whole point of an additive model is that you can look at each feature's
effect in isolation. This module turns the trained subnetworks into
actuarial-friendly outputs:

1. Plot shape curves (equivalent to GLM marginal effect plots)
2. Export to Polars DataFrames for tabular review
3. Export to JSON for actuarial documentation systems
4. Compare shape functions to GLM-equivalent relativities

Shape function plotting is intentionally kept simple — matplotlib only, no
interactive dependencies. Actuaries use these plots in Word/Excel reports,
so static PNG output is what they actually need.

For categorical features, the output is a bar chart of category contributions.
For continuous features, it's a line plot of f_i(x) over the observed range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch


@dataclass
class ShapeFunction:
    """Extracted shape function for one feature.

    Contains the evaluated curve (x_values, f_values) and metadata for
    reporting. Created by ANAM.shape_functions() after fitting.

    Attributes
    ----------
    feature_name:
        Feature identifier.
    feature_type:
        'continuous' or 'categorical'.
    x_values:
        Grid of feature values (continuous) or category indices (categorical).
    f_values:
        Corresponding subnetwork outputs f_i(x_i).
    x_label:
        Human-readable x-axis label for plots.
    monotonicity:
        Monotonicity constraint applied during training.
    category_labels:
        Optional mapping from category index to label string.
    """

    feature_name: str
    feature_type: str
    x_values: np.ndarray
    f_values: np.ndarray
    x_label: str = ""
    monotonicity: str = "none"
    category_labels: Optional[Dict[int, str]] = None

    def to_polars(self) -> pl.DataFrame:
        """Export shape function as a Polars DataFrame.

        For continuous features: columns [x, f_x]
        For categorical features: columns [category_index, category_label, f_x]
        """
        if self.feature_type == "continuous":
            return pl.DataFrame(
                {
                    "x": self.x_values.tolist(),
                    "f_x": self.f_values.tolist(),
                    "feature": [self.feature_name] * len(self.x_values),
                }
            )
        else:
            labels = [
                self.category_labels.get(int(i), str(int(i)))
                if self.category_labels
                else str(int(i))
                for i in self.x_values
            ]
            return pl.DataFrame(
                {
                    "category_index": self.x_values.astype(int).tolist(),
                    "category_label": labels,
                    "f_x": self.f_values.tolist(),
                    "feature": [self.feature_name] * len(self.x_values),
                }
            )

    def to_relativities(self, base_level: Optional[float] = None) -> pl.DataFrame:
        """Convert shape function to GLM-style multiplicative relativities.

        For log-link models, the shape function is on the log scale.
        exp(f_i(x_i)) gives the multiplicative factor. This is then
        normalised by the value at the base level (default: median x).

        Parameters
        ----------
        base_level:
            Feature value to use as base (relativity = 1.0). For
            continuous features, defaults to median. For categorical,
            defaults to category with f_x closest to zero.

        Returns
        -------
        pl.DataFrame
            Columns: [x, relativity, log_relativity]
        """
        if self.feature_type == "continuous":
            if base_level is None:
                mid_idx = len(self.x_values) // 2
                base_f = self.f_values[mid_idx]
            else:
                # Find nearest grid point
                idx = np.argmin(np.abs(self.x_values - base_level))
                base_f = self.f_values[idx]

            log_rel = self.f_values - base_f
            rel = np.exp(log_rel)

            return pl.DataFrame(
                {
                    "x": self.x_values.tolist(),
                    "relativity": rel.tolist(),
                    "log_relativity": log_rel.tolist(),
                    "feature": [self.feature_name] * len(self.x_values),
                }
            )
        else:
            if base_level is None:
                base_f = self.f_values[np.argmin(np.abs(self.f_values))]
            else:
                base_f = self.f_values[int(base_level)]

            log_rel = self.f_values - base_f
            rel = np.exp(log_rel)
            labels = [
                self.category_labels.get(int(i), str(int(i)))
                if self.category_labels
                else str(int(i))
                for i in self.x_values
            ]

            return pl.DataFrame(
                {
                    "category_index": self.x_values.astype(int).tolist(),
                    "category_label": labels,
                    "relativity": rel.tolist(),
                    "log_relativity": log_rel.tolist(),
                    "feature": [self.feature_name] * len(self.x_values),
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "monotonicity": self.monotonicity,
            "x_values": self.x_values.tolist(),
            "f_values": self.f_values.tolist(),
            "x_label": self.x_label,
            "category_labels": self.category_labels,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def plot(
        self,
        ax: Optional[Any] = None,
        show_monotonicity: bool = True,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 4),
    ) -> Any:
        """Plot the shape function.

        Parameters
        ----------
        ax:
            Matplotlib axes object. Creates a new figure if None.
        show_monotonicity:
            Annotate the plot with the monotonicity constraint.
        title:
            Plot title. Defaults to feature name.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if self.feature_type == "continuous":
            ax.plot(self.x_values, self.f_values, linewidth=2, color="#2c6fad")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.fill_between(
                self.x_values, self.f_values, 0,
                alpha=0.15, color="#2c6fad"
            )
            ax.set_xlabel(self.x_label or self.feature_name)
            ax.set_ylabel("log contribution")

            if show_monotonicity and self.monotonicity != "none":
                mono_label = f"monotone {self.monotonicity}"
                ax.annotate(
                    mono_label,
                    xy=(0.02, 0.95),
                    xycoords="axes fraction",
                    fontsize=9,
                    color="darkgreen",
                    va="top",
                )

        else:
            # Categorical: bar chart
            labels = [
                self.category_labels.get(int(i), str(int(i)))
                if self.category_labels
                else str(int(i))
                for i in self.x_values
            ]
            colors = [
                "#d63031" if v < 0 else "#0984e3" for v in self.f_values
            ]
            ax.bar(labels, self.f_values, color=colors, edgecolor="white")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel(self.x_label or self.feature_name)
            ax.set_ylabel("log contribution")
            ax.tick_params(axis="x", rotation=45)

        ax.set_title(title or f"Shape function: {self.feature_name}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax


def extract_shape_functions(
    model: "ANAMModel",
    X_train: np.ndarray,
    n_points: int = 200,
    category_labels: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[str, ShapeFunction]:
    """Extract all shape functions from a trained model.

    Evaluates each subnetwork over the observed range of the training data.

    Parameters
    ----------
    model:
        Trained ANAMModel.
    X_train:
        Training feature matrix. Used to determine observed feature ranges
        and category sets.
    n_points:
        Number of grid points for continuous features.
    category_labels:
        Optional dict mapping feature_name -> {category_idx -> label string}.

    Returns
    -------
    Dict[str, ShapeFunction]
        Feature name -> ShapeFunction for each feature.
    """
    shapes: Dict[str, ShapeFunction] = {}

    model.eval()
    with torch.no_grad():
        for i, cfg in enumerate(model.feature_configs):
            col = X_train[:, i]
            net = model.feature_nets[cfg.name]
            cat_labels = (
                category_labels.get(cfg.name) if category_labels else None
            )

            if cfg.feature_type == "continuous":
                x_min, x_max = float(col.min()), float(col.max())
                x_grid = torch.linspace(x_min, x_max, n_points)
                f_vals = net(x_grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                x_vals = x_grid.cpu().numpy()

            else:
                unique_cats = np.unique(col.astype(int))
                x_cat = torch.tensor(unique_cats, dtype=torch.long)
                f_vals = net(x_cat).squeeze(-1).cpu().numpy()
                x_vals = unique_cats.astype(float)

            shapes[cfg.name] = ShapeFunction(
                feature_name=cfg.name,
                feature_type=cfg.feature_type,
                x_values=x_vals,
                f_values=f_vals,
                x_label=cfg.name,
                monotonicity=cfg.monotonicity if hasattr(cfg, "monotonicity") else "none",
                category_labels=cat_labels,
            )

    return shapes


def plot_all_shapes(
    shapes: Dict[str, ShapeFunction],
    n_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (5, 3),
    suptitle: str = "ANAM Shape Functions",
) -> Any:
    """Plot all shape functions in a grid layout.

    Parameters
    ----------
    shapes:
        Dict returned by extract_shape_functions().
    n_cols:
        Number of columns in the subplot grid.
    figsize_per_plot:
        Width and height per subplot panel.
    suptitle:
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n = len(shapes)
    n_rows = (n + n_cols - 1) // n_cols
    fig_w = figsize_per_plot[0] * n_cols
    fig_h = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    for ax, (name, sf) in zip(axes_flat, shapes.items()):
        sf.plot(ax=ax)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
