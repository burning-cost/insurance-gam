"""
Tests for insurance_gam.post_selection.

Synthetic DGP: Poisson, n=2000, p=10, 3 true features (x0, x1, x2),
7 noise features (x3-x9).

    log(mu_i) = 0.4*x0 + 0.3*x1 + 0.2*x2

Tests cover:
  - Selection recovers true features
  - PSI CIs cover true coefficients (single-trial coverage check)
  - Exposure offset passes through correctly
  - forest_plot runs without error
  - DataSplitPostSelectionGLM produces valid Wald CIs
  - Input validation (family, alpha, negative y)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_gam.post_selection import (
    DataSplitPostSelectionGLM,
    PostSelectionGLM,
)

# ---------------------------------------------------------------------------
# Shared DGP fixture
# ---------------------------------------------------------------------------

TRUE_COEFS = np.array([0.4, 0.3, 0.2])  # x0, x1, x2
N = 2000
P = 10
SEED = 42


@pytest.fixture(scope="module")
def poisson_data():
    """Synthetic Poisson dataset: n=2000, p=10, 3 true features."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, P))
    eta = X[:, :3] @ TRUE_COEFS
    y = rng.poisson(np.exp(eta))
    return X, y


@pytest.fixture(scope="module")
def poisson_data_with_exposure():
    """Synthetic Poisson rate dataset with exposure."""
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((N, P))
    exposure = rng.uniform(0.5, 2.0, size=N)
    eta = X[:, :3] @ TRUE_COEFS + np.log(exposure)
    y = rng.poisson(np.exp(eta))
    return X, y, exposure


# ---------------------------------------------------------------------------
# PostSelectionGLM tests
# ---------------------------------------------------------------------------


class TestPostSelectionGLM:
    def test_fit_returns_self(self, poisson_data):
        X, y = poisson_data
        model = PostSelectionGLM(random_state=SEED)
        result = model.fit(X, y)
        assert result is model

    def test_summary_returns_dataframe(self, poisson_data):
        X, y = poisson_data
        model = PostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "feature", "coefficient", "std_error",
            "ci_lower", "ci_upper", "pvalue", "selected",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == P

    def test_selection_includes_true_features(self, poisson_data):
        """Lasso should recover x0, x1, x2 (signal > noise at n=2000)."""
        X, y = poisson_data
        model = PostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        selected_names = set(df[df["selected"]]["feature"].tolist())
        true_names = {"x0", "x1", "x2"}
        assert true_names.issubset(selected_names), (
            f"True features not fully recovered. Selected: {selected_names}"
        )

    def test_psi_cis_cover_true_coefficients(self, poisson_data):
        """PSI CIs for selected features should contain the true values."""
        X, y = poisson_data
        model = PostSelectionGLM(alpha=0.05, random_state=SEED).fit(X, y)
        df = model.summary()
        for i, (feat, true_coef) in enumerate(
            zip(["x0", "x1", "x2"], TRUE_COEFS)
        ):
            row = df[df["feature"] == feat]
            assert not row.empty
            if not row["selected"].values[0]:
                continue  # Skip if not selected (handled by selection test)
            lo = row["ci_lower"].values[0]
            hi = row["ci_upper"].values[0]
            assert not np.isnan(lo), f"CI lower NaN for {feat}"
            assert not np.isnan(hi), f"CI upper NaN for {feat}"
            assert lo < hi, f"CI inverted for {feat}: [{lo}, {hi}]"
            # Coverage check: true coefficient should be inside the CI
            assert lo <= true_coef <= hi, (
                f"True coef {true_coef:.3f} for {feat} outside CI "
                f"[{lo:.3f}, {hi:.3f}]"
            )

    def test_coefficients_close_to_truth(self, poisson_data):
        """MLE estimates should be within 0.2 of true values."""
        X, y = poisson_data
        model = PostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        for feat, true_coef in zip(["x0", "x1", "x2"], TRUE_COEFS):
            row = df[df["feature"] == feat]
            if row["selected"].values[0]:
                coef = row["coefficient"].values[0]
                assert abs(coef - true_coef) < 0.20, (
                    f"Estimate {coef:.3f} too far from truth {true_coef:.3f} "
                    f"for {feat}"
                )

    def test_exposure_offset_works(self, poisson_data_with_exposure):
        """Model with exposure should converge and produce valid CIs."""
        X, y, exposure = poisson_data_with_exposure
        model = PostSelectionGLM(random_state=SEED).fit(X, y, exposure=exposure)
        df = model.summary()
        selected = df[df["selected"]]
        assert len(selected) >= 1
        assert not selected["coefficient"].isna().all()

    def test_exposure_selects_true_features(self, poisson_data_with_exposure):
        """With exposure, x0/x1/x2 should still be selected."""
        X, y, exposure = poisson_data_with_exposure
        model = PostSelectionGLM(random_state=SEED).fit(X, y, exposure=exposure)
        df = model.summary()
        selected_names = set(df[df["selected"]]["feature"].tolist())
        true_names = {"x0", "x1", "x2"}
        assert true_names.issubset(selected_names), (
            f"True features not recovered with exposure. "
            f"Selected: {selected_names}"
        )

    def test_forest_plot_runs(self, poisson_data):
        """forest_plot should return Axes without raising."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = poisson_data
        model = PostSelectionGLM(random_state=SEED).fit(X, y)
        fig, ax = plt.subplots()
        returned_ax = model.forest_plot(ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_dataframe_input_preserves_column_names(self, poisson_data):
        """DataFrame input should use column names as feature_names_."""
        X, y = poisson_data
        col_names = [f"feat_{i}" for i in range(P)]
        X_df = pd.DataFrame(X, columns=col_names)
        model = PostSelectionGLM(random_state=SEED).fit(X_df, y)
        df = model.summary()
        assert set(df["feature"]) == set(col_names)

    def test_summary_before_fit_raises(self):
        model = PostSelectionGLM()
        with pytest.raises(RuntimeError, match="fit()"):
            model.summary()

    def test_forest_plot_before_fit_raises(self):
        model = PostSelectionGLM()
        with pytest.raises(RuntimeError, match="fit()"):
            model.forest_plot()

    def test_max_features_constraint(self, poisson_data):
        """max_features=2 should select at most 2 features."""
        X, y = poisson_data
        model = PostSelectionGLM(max_features=2, random_state=SEED).fit(X, y)
        df = model.summary()
        assert df["selected"].sum() <= 2

    def test_ci_ordering(self, poisson_data):
        """ci_lower <= coefficient <= ci_upper for all selected features."""
        X, y = poisson_data
        model = PostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()[model.summary()["selected"]]
        for _, row in df.iterrows():
            if not np.isnan(row["ci_lower"]):
                assert row["ci_lower"] <= row["ci_upper"], (
                    f"CI inverted for {row['feature']}"
                )


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_unsupported_family_raises(self):
        with pytest.raises(ValueError, match="poisson"):
            PostSelectionGLM(family="gamma")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            PostSelectionGLM(alpha=1.5)

    def test_negative_y_raises(self, poisson_data):
        X, y = poisson_data
        y_bad = y.copy()
        y_bad[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            PostSelectionGLM().fit(X, y_bad)

    def test_zero_exposure_raises(self, poisson_data):
        X, y = poisson_data
        exposure = np.ones(N)
        exposure[5] = 0.0
        with pytest.raises(ValueError, match="positive"):
            PostSelectionGLM().fit(X, y, exposure=exposure)

    def test_wrong_y_shape_raises(self, poisson_data):
        X, y = poisson_data
        with pytest.raises(ValueError):
            PostSelectionGLM().fit(X, y[:100])

    def test_datasplit_unsupported_family_raises(self):
        with pytest.raises(ValueError, match="poisson"):
            DataSplitPostSelectionGLM(family="gamma")


# ---------------------------------------------------------------------------
# DataSplitPostSelectionGLM tests
# ---------------------------------------------------------------------------


class TestDataSplitPostSelectionGLM:
    def test_fit_returns_self(self, poisson_data):
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED)
        result = model.fit(X, y)
        assert result is model

    def test_summary_columns(self, poisson_data):
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        expected = {
            "feature", "coefficient", "std_error",
            "ci_lower", "ci_upper", "pvalue", "selected",
        }
        assert expected.issubset(set(df.columns))
        assert len(df) == P

    def test_selection_includes_true_features(self, poisson_data):
        """Data-split Lasso should recover x0, x1, x2."""
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        selected_names = set(df[df["selected"]]["feature"].tolist())
        true_names = {"x0", "x1", "x2"}
        assert true_names.issubset(selected_names), (
            f"True features not recovered. Selected: {selected_names}"
        )

    def test_wald_cis_cover_true_coefficients(self, poisson_data):
        """Wald CIs on inference half should cover true coefficients."""
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(alpha=0.05, random_state=SEED).fit(
            X, y
        )
        df = model.summary()
        for feat, true_coef in zip(["x0", "x1", "x2"], TRUE_COEFS):
            row = df[df["feature"] == feat]
            if not row["selected"].values[0]:
                continue
            lo = row["ci_lower"].values[0]
            hi = row["ci_upper"].values[0]
            assert lo < hi
            assert lo <= true_coef <= hi, (
                f"True coef {true_coef:.3f} for {feat} outside Wald CI "
                f"[{lo:.3f}, {hi:.3f}]"
            )

    def test_valid_cis_for_all_selected(self, poisson_data):
        """All selected features should have non-NaN, non-inverted CIs."""
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()[model.summary()["selected"]]
        assert len(df) >= 1
        assert not df["ci_lower"].isna().any()
        assert not df["ci_upper"].isna().any()
        assert (df["ci_lower"] < df["ci_upper"]).all()

    def test_pvalues_small_for_true_features(self, poisson_data):
        """True features should have p-values < 0.05."""
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        for feat in ["x0", "x1", "x2"]:
            row = df[df["feature"] == feat]
            if row["selected"].values[0]:
                pval = row["pvalue"].values[0]
                assert pval < 0.05, (
                    f"p-value {pval:.4f} not significant for {feat}"
                )

    def test_exposure_offset_works(self, poisson_data_with_exposure):
        """Data-split model should work with exposure."""
        X, y, exposure = poisson_data_with_exposure
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(
            X, y, exposure=exposure
        )
        df = model.summary()
        selected = df[df["selected"]]
        assert len(selected) >= 1

    def test_forest_plot_runs(self, poisson_data):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        ax = model.forest_plot()
        assert ax is not None
        plt.close("all")

    def test_summary_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit()"):
            DataSplitPostSelectionGLM().summary()

    def test_non_selected_have_nan_cis(self, poisson_data):
        """Features not selected should have NaN coefficients."""
        X, y = poisson_data
        model = DataSplitPostSelectionGLM(random_state=SEED).fit(X, y)
        df = model.summary()
        non_sel = df[~df["selected"]]
        assert non_sel["coefficient"].isna().all()
        assert non_sel["ci_lower"].isna().all()
