# Changelog

## [0.2.0] - 2026-03-31

### Added
- `EBMInsuranceWrapper` — unified actuarial facade for `InsuranceEBM`. Consolidates
  the full actuarial workflow (fitting, prediction, diagnostics, relativities) into
  a single object and adds four capabilities not previously available:

  1. **Balance check and correction** — `balance_check()` returns global A/E ratio
     (sum actual / sum predicted). `apply_balance_correction()` shifts the EBM
     intercept by log(A/E) to enforce exact global balance, preserving all shape
     functions. Module-level `balance_ratio()` function also exported for standalone use.

  2. **Interaction heatmap** — `interaction_heatmap()` extracts 2-D shape functions
     for EBM pairwise interaction terms and returns them as a long-format polars
     DataFrame (bin_i, bin_j, score, relativity). `interaction_heatmap_wide()` pivots
     this to a wide table for Excel export or notebook display.

  3. **EU AI Act Article 13 transparency report** — `transparency_report()` generates
     a JSON-serialisable dict with model metadata, performance metrics (deviance, Gini,
     balance ratio), feature profiles (relativity ranges), interaction terms, and a
     structured Article 13 checklist marking what is provided vs what requires human
     input. `transparency_report_json()` returns the same as a formatted JSON string.

  4. **Exposure-weighted k-fold cross-validation** — `cross_validate()` method on the
     wrapper (and module-level `cross_validate()` function) runs k-fold CV on any
     `InsuranceEBM` configuration, returning per-fold deviances plus mean and std.
     Each fold trains a fresh model — no warm-starting.

- Convenience methods on `EBMInsuranceWrapper`: `ae_by_decile()`, `gini_coefficient()`,
  `segment_ae_table()`, `relativity_table()`, `relativity_summary()` — thin delegators
  to the existing diagnostics modules, so you do not need to import them separately.

- New exports from `insurance_gam.ebm`: `EBMInsuranceWrapper`, `balance_ratio`,
  `cross_validate`.

- 59 new tests in `tests/ebm/test_wrapper.py`.

## [0.1.9] - 2026-03-27

### Fixed
- Fixed test assertions in `test_helpful_import_errors.py` that checked for
  `pip install insurance-gam[anam]` and `pip install insurance-gam[pin]` — the
  actual error messages correctly say `pip install insurance-gam[neural]`. The
  tests would have failed on every CI run. (First-run UX audit, March 2026.)
- Replaced deprecated `pl.Utf8` with `pl.String` in `ebm/_diagnostics.py` and
  `anam/utils.py`. `pl.Utf8` is a legacy alias that Polars may remove in a
  future major version. Both files now use the canonical `pl.String` type.
- Fixed README installation note where package names were accidentally stripped,
  producing the text "The three subpackages are independent:  loads interpretML,
   loads PyTorch." — restored to name the subpackages explicitly.

## [0.1.8] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)



## v0.1.6 (2026-03-22) [unreleased]
- Add Databricks benchmark script and update benchmark results section
- fix: move flask/werkzeug/pyasn1 from core deps to [ebm] optional extra
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.6 (2026-03-21)
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- Add pdoc API documentation with GitHub Pages
- docs: regenerate API reference [skip ci]
- bump: v0.1.6 — helpful ImportError for missing extras
- docs: regenerate API reference [skip ci]
- Add helpful ImportError for missing extras in gam subpackages
- security: pin transitive deps to close 5 Dependabot alerts
- fix: README technical errors from quality review
- Add MIT license
- docs: regenerate API reference [skip ci]
- Fix P0/P1/P2 quality audit issues
- Add PyPI classifiers for financial/insurance audience
- Add Colab quickstart notebook and Open in Colab badge
- benchmarks: Databricks run of EBM vs GLM benchmark, honest results
- refresh benchmark numbers post-P0 fixes
- docs: regenerate API reference [skip ci]
- fix: resolve P0 and P1 bugs from code review (v0.1.3)
- docs: regenerate API reference [skip ci]
- Add benchmark: InsuranceEBM vs Poisson GLM on non-linear frequency DGP
- pin statsmodels>=0.14.5 for scipy compat
- v0.1.2: Support 'Nx' multiplier strings for interactions parameter
- Add shields.io badge row (PyPI, Python, Tests, License)
- Add Quick Start section to README
- docs: add Databricks notebook link
- fix: handle list and ndarray for feature_names_in_ in RelativitiesTable
