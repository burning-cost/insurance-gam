"""
Tests for RelativitiesTable: relativity calculation, base level selection,
summary table, and Excel export.
"""

import numpy as np
import polars as pl
import pytest

from insurance_gam.ebm import RelativitiesTable


class TestRelativitiesTable:
    def test_construction(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        assert rt.model is fitted_poisson_model

    def test_table_returns_polars_dataframe(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        assert isinstance(df, pl.DataFrame)

    def test_table_columns(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        assert set(df.columns) == {"bin_label", "raw_score", "relativity"}

    def test_base_bin_has_relativity_one(self, fitted_poisson_model):
        """The modal bin should have relativity exactly 1.0."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        rels = df["relativity"].to_numpy()
        # At least one bin should have relativity = 1.0 (the base)
        assert np.any(np.isclose(rels, 1.0, atol=1e-6)), (
            f"Expected at least one bin with relativity=1.0, got min={rels.min():.4f}"
        )

    def test_relativities_positive(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        assert np.all(df["relativity"].to_numpy() > 0)

    def test_categorical_feature(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("vehicle_group")
        assert len(df) > 0
        assert np.all(df["relativity"].to_numpy() > 0)

    def test_relativity_exp_of_score_diff(self, fitted_poisson_model):
        """Verify relativities = exp(score - base_score) manually."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("ncd")
        raw = df["raw_score"].to_numpy()
        rels = df["relativity"].to_numpy()

        # Find base bin (closest to 0 diff, i.e. where relativity = 1)
        base_idx = np.argmin(np.abs(rels - 1.0))
        base_score = raw[base_idx]

        expected_rels = np.exp(raw - base_score)
        np.testing.assert_allclose(rels, expected_rels, rtol=1e-5)

    def test_invalid_feature_raises(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ValueError, match="not found"):
            rt.table("nonexistent_feature")

    def test_summary_returns_polars_dataframe(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        assert isinstance(df, pl.DataFrame)

    def test_summary_columns(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        expected = {"feature", "n_bins", "min_relativity", "max_relativity", "range"}
        assert expected.issubset(set(df.columns))

    def test_summary_has_all_features(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        # Should have at least the main features (may exclude interactions)
        assert len(df) >= 1

    def test_summary_range_geq_one(self, fitted_poisson_model):
        """range = max/min relativity >= 1 by definition."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        ranges = df["range"].to_numpy()
        assert np.all(ranges >= 1.0 - 1e-9)

    def test_summary_sorted_by_range(self, fitted_poisson_model):
        """Summary should be sorted by range descending."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        ranges = df["range"].to_numpy()
        assert np.all(np.diff(ranges) <= 0), "Summary should be sorted by range descending"

    def test_n_bins_positive(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.summary()
        assert np.all(df["n_bins"].to_numpy() > 0)


class TestPlot:
    def test_plot_bar(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")
        rt = RelativitiesTable(fitted_poisson_model)
        ax = rt.plot("driver_age", kind="bar")
        assert ax is not None

    def test_plot_line(self, fitted_poisson_model):
        import matplotlib
        matplotlib.use("Agg")
        rt = RelativitiesTable(fitted_poisson_model)
        ax = rt.plot("ncd", kind="line")
        assert ax is not None

    def test_plot_invalid_kind_raises(self, fitted_poisson_model):
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ValueError, match="kind"):
            rt.plot("ncd", kind="scatter")


class TestExcelExport:
    def test_export_excel(self, fitted_poisson_model, tmp_path):
        pytest.importorskip("openpyxl")
        rt = RelativitiesTable(fitted_poisson_model)
        path = tmp_path / "relativities.xlsx"
        rt.export_excel(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_excel_no_openpyxl(self, fitted_poisson_model, tmp_path, monkeypatch):
        """Should raise ImportError if openpyxl is not available."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("openpyxl not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        rt = RelativitiesTable(fitted_poisson_model)
        with pytest.raises(ImportError, match="openpyxl"):
            rt.export_excel(tmp_path / "test.xlsx")


class TestP01ModalBinIdxRegression:
    """Regression tests for P0-1: _modal_bin_idx sliced wrong end of weights array.

    interpretML places the missing-value bin at index 0, not the end.
    The fix is weights[1:] (skip index-0 missing bin) not weights[:-1].
    """

    def test_modal_bin_skips_index_zero_not_last(self, fitted_poisson_model):
        """Modal bin index must correspond to the highest-weight real bin.

        The mock EBM has weights [5.0, 20.0, 25.0, 30.0, 20.0, 10.0] where
        index 0 is the missing bin (weight=5.0). The modal real bin is at
        index 3 (weight=30.0) within the full array, i.e. index 2 within
        weights[1:]. After argmax over weights[1:] = [20,25,30,20,10],
        argmax=2, so the returned index should be 3 (= 2 + 1 offset when
        RelativitiesTable clips to the scores array, which has 5 elements
        from explain_global returning scores[1:]).

        The scores array from the mock has 5 bins (explain_global skips the
        missing bin in the name/scores it returns), so base_idx must be
        in 0..4 and point to the bin with weight 30.0.
        """
        from insurance_gam.ebm._relativities import _modal_bin_idx
        ebm = fitted_poisson_model.ebm_

        # weights[1:] = [20, 25, 30, 20, 10] -> argmax = 2
        # weights[:-1] = [5, 20, 25, 30, 20] -> argmax = 3 (old, wrong result)
        idx = _modal_bin_idx(ebm, "driver_age")
        assert idx == 2, (
            f"Expected modal bin index 2 (argmax of weights[1:]), got {idx}. "
            "This suggests the old weights[:-1] bug is still present."
        )

    def test_base_bin_has_relativity_one_after_fix(self, fitted_poisson_model):
        """The base bin (highest weight) must have relativity == 1.0."""
        rt = RelativitiesTable(fitted_poisson_model)
        df = rt.table("driver_age")
        rels = df["relativity"].to_numpy()
        assert np.any(np.isclose(rels, 1.0, atol=1e-6)), (
            "No bin has relativity=1.0 — base bin selection is broken."
        )

    def test_missing_bin_not_chosen_as_base(self, fitted_poisson_model):
        """Index 0 of weights (missing-value bin) must never be the base.

        Before the fix, weights[:-1] excluded the last bin. With the interpretML
        convention of missing bin at index 0, this meant index 0 (missing) was
        included in argmax, and a low-frequency missing bin could be chosen as
        the base level — wrong for a rating table.
        """
        from insurance_gam.ebm._relativities import _modal_bin_idx
        ebm = fitted_poisson_model.ebm_
        # For our mock, weights[0]=5.0 is the lowest weight.
        # weights[1:] = [20,25,30,20,10] -> argmax=2 (real bin index 2)
        # If index 0 were returned, that would mean the missing bin was chosen.
        idx = _modal_bin_idx(ebm, "driver_age")
        # The returned index is used to index into the scores array (len 5).
        # Index 0 of the scores array corresponds to weights[1] (the first real bin).
        # weights[0] is missing bin; if idx==0, scores[0] is real bin 1 -- acceptable.
        # But the bug was that the OLD code could return index 3 (the 30.0 bin under
        # weights[:-1]) vs index 2 (the 30.0 bin under weights[1:]).
        # The modal bin weight is 30.0 at weights[3]; in the scores array this maps to
        # scores index = 3 - 1 = 2 (explain_global returns scores[1:]).
        assert idx == 2, f"Expected 2, got {idx}"
