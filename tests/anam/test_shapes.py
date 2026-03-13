"""
test_shapes.py — Tests for shape function extraction, plotting, and export.
"""

import numpy as np
import pytest
import polars as pl

from insurance_gam.anam.shapes import (
    ShapeFunction,
    extract_shape_functions,
    plot_all_shapes,
)


def make_continuous_shape():
    x = np.linspace(0.0, 10.0, 50)
    f = np.sin(x)
    return ShapeFunction(
        feature_name="test_feature",
        feature_type="continuous",
        x_values=x,
        f_values=f,
        x_label="Test Feature",
        monotonicity="none",
    )


def make_categorical_shape():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    f = np.array([0.0, 0.3, -0.1, 0.5])
    labels = {0: "region_a", 1: "region_b", 2: "region_c", 3: "region_d"}
    return ShapeFunction(
        feature_name="region",
        feature_type="categorical",
        x_values=x,
        f_values=f,
        x_label="Region",
        category_labels=labels,
    )


class TestShapeFunction:
    def test_to_polars_continuous(self):
        sf = make_continuous_shape()
        df = sf.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "x" in df.columns
        assert "f_x" in df.columns
        assert len(df) == 50

    def test_to_polars_categorical(self):
        sf = make_categorical_shape()
        df = sf.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "category_index" in df.columns
        assert "category_label" in df.columns
        assert "f_x" in df.columns
        assert len(df) == 4

    def test_to_polars_categorical_labels(self):
        sf = make_categorical_shape()
        df = sf.to_polars()
        labels = df["category_label"].to_list()
        assert "region_a" in labels
        assert "region_d" in labels

    def test_to_relativities_continuous(self):
        sf = make_continuous_shape()
        df = sf.to_relativities()
        assert "relativity" in df.columns
        assert "log_relativity" in df.columns
        assert len(df) == 50
        # Relativities should be positive
        assert (df["relativity"].to_numpy() > 0).all()

    def test_to_relativities_categorical(self):
        sf = make_categorical_shape()
        df = sf.to_relativities()
        assert "relativity" in df.columns
        assert len(df) == 4
        assert (df["relativity"].to_numpy() > 0).all()

    def test_to_dict_contains_keys(self):
        sf = make_continuous_shape()
        d = sf.to_dict()
        assert "feature_name" in d
        assert "feature_type" in d
        assert "x_values" in d
        assert "f_values" in d

    def test_to_json_is_valid_json(self):
        import json
        sf = make_continuous_shape()
        json_str = sf.to_json()
        parsed = json.loads(json_str)
        assert parsed["feature_name"] == "test_feature"

    def test_plot_continuous_no_error(self):
        """Plotting should not raise for continuous features."""
        import matplotlib
        matplotlib.use("Agg")
        sf = make_continuous_shape()
        ax = sf.plot()
        assert ax is not None

    def test_plot_categorical_no_error(self):
        """Plotting should not raise for categorical features."""
        import matplotlib
        matplotlib.use("Agg")
        sf = make_categorical_shape()
        ax = sf.plot()
        assert ax is not None

    def test_relativity_base_at_median(self):
        """Default base level should put median relativity at 1.0."""
        sf = make_continuous_shape()
        df = sf.to_relativities()
        mid_idx = len(sf.x_values) // 2
        mid_rel = df["relativity"].to_numpy()[mid_idx]
        assert mid_rel == pytest.approx(1.0, rel=1e-3)


class TestExtractShapeFunctions:
    def test_extract_returns_all_features(self, trained_anam, synthetic_data):
        """extract_shape_functions should return one ShapeFunction per feature."""
        shapes = extract_shape_functions(
            trained_anam, synthetic_data["X"], n_points=50
        )
        feature_names = {"driver_age", "vehicle_age", "ncd", "region", "vehicle_type"}
        assert set(shapes.keys()) == feature_names

    def test_continuous_shape_n_points(self, trained_anam, synthetic_data):
        """Continuous shape functions should have n_points values."""
        shapes = extract_shape_functions(
            trained_anam, synthetic_data["X"], n_points=100
        )
        assert len(shapes["driver_age"].x_values) == 100
        assert len(shapes["driver_age"].f_values) == 100

    def test_categorical_shape_has_all_categories(self, trained_anam, synthetic_data):
        """Categorical shape should have one value per observed category."""
        shapes = extract_shape_functions(
            trained_anam, synthetic_data["X"], n_points=100
        )
        region_shape = shapes["region"]
        # Region has 4 categories (0, 1, 2, 3)
        assert len(region_shape.x_values) == 4

    def test_shape_values_finite(self, trained_anam, synthetic_data):
        """All shape function values should be finite."""
        shapes = extract_shape_functions(
            trained_anam, synthetic_data["X"], n_points=50
        )
        for name, sf in shapes.items():
            assert np.isfinite(sf.f_values).all(), f"Non-finite values in {name}"

    def test_shape_with_category_labels(self, trained_anam, synthetic_data):
        """Category labels should be attached when provided."""
        labels = {"region": {0: "London", 1: "North", 2: "Midlands", 3: "Scotland"}}
        shapes = extract_shape_functions(
            trained_anam, synthetic_data["X"], n_points=50,
            category_labels=labels
        )
        sf = shapes["region"]
        assert sf.category_labels is not None
        assert sf.category_labels[0] == "London"


class TestPlotAllShapes:
    def test_plot_all_returns_figure(self, trained_anam, synthetic_data):
        """plot_all_shapes should return a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shapes = extract_shape_functions(
            trained_anam, synthetic_data["X"], n_points=30
        )
        fig = plot_all_shapes(shapes)
        assert fig is not None
        plt.close("all")
