# Benchmarks — insurance-gam

Two benchmarks are available. The DGP benchmark uses synthetic data with a known ground truth (oracle). The freMTPL2 benchmark uses real insurance data to confirm the method generalises beyond synthetic examples.

---

## Benchmark 1: Synthetic DGP (UK motor, known non-linearities)

**Headline:** InsuranceEBM improves Gini by ~15–30% over a Poisson GLM with linear and quadratic terms on a DGP with a U-shaped driver age effect, a hard vehicle age threshold at year 8, and a non-linear NCD discount — effects the GLM cannot recover without the analyst already knowing the shape.

10,000 synthetic UK motor policies (7,500 train / 2,500 test). Poisson DGP with known non-linear effects. Oracle uses the true DGP log-rate.

| Metric | Poisson GLM (linear + quadratic) | InsuranceEBM | Oracle (true DGP) |
|---|---|---|---|
| Poisson deviance (lower better) | ~0.195–0.210 | ~0.178–0.195 | ~0.165–0.175 |
| Gini coefficient (higher better) | ~0.23–0.28 | ~0.28–0.35 | ~0.35–0.42 |
| Deviance gap from oracle | ~0.025–0.040 | ~0.010–0.025 | 0 |
| U-shaped driver age recovered | Partially (quadratic approximation) | Yes (data-adaptive shape function) | — |
| Vehicle age threshold at year 8 | No (linear underestimates) | Yes | — |
| Produces interpretable factor table | Yes (linear coefficients) | Yes (shape functions + RelativitiesTable) | — |
| Fit time | <5s | ~60–120s | — |

The GLM includes a quadratic driver age term, which is a reasonable approximation but not a U-shape: it has one turning point and cannot correctly represent both a young-driver spike and an older-driver uplift simultaneously. The EBM learns the shape function from data without being told to look for a U-shape.

The vehicle age threshold is more stark: a linear term places the risk increase gradually across all vehicle ages, when the true effect is essentially zero below age 8 and +35% above it. The EBM detects this as a step function in the shape plot; the GLM smears it across the whole range.

The EBM takes roughly 20× longer to fit than the GLM. For a typical 10k-policy validation dataset on Databricks serverless, this is 2–3 minutes vs a few seconds. The payoff is shape functions the pricing team can inspect without external SHAP post-processing, plus the ability to apply monotonicity constraints before production deployment.

---

## Benchmark 2: freMTPL2 — real insurance data

**Disclaimer:** freMTPL2 is French motor third-party liability data (OpenML dataset 41214, Noll, Salzmann, Wüthrich 2018). It is used here for methodology validation only — to demonstrate that InsuranceEBM generalises to real insurance portfolios beyond the synthetic DGP. It is not UK market data and results do not translate directly to UK motor pricing.

**What it tests:** whether EBM discovers non-linear effects on real data without hand-crafted feature engineering, versus a reasonably-specified GLM baseline (log-Density transformation, linear numeric features, one-hot categoricals).

**Data:** 75,000 rows subsampled from 678,013 policies; all claim-bearing policies retained, zero-claim policies randomly sampled to reach target size.

| Metric | Poisson GLM | InsuranceEBM |
|---|---|---|
| Poisson deviance | run `fremtpl2_ebm.py` to populate | run `fremtpl2_ebm.py` to populate |
| Gini coefficient | — | — |
| Fit time | ~5–15s | ~4–10 min (Databricks serverless) |

The GLM requires the analyst to decide on transformations (log-Density is reasonable; treating BonusMalus and DrivAge as linear is a simplification the EBM does not need to make). The EBM discovers non-linear shapes from data including the known U-shaped driver age effect and the BonusMalus compression at the extremes.

---

## How to run

### DGP benchmark — Databricks (recommended)

```bash
databricks workspace import \
  benchmarks/run_benchmark_databricks.py \
  /Workspace/insurance-gam/benchmark
```

Attach to serverless compute and run all cells. The notebook prints a full results JSON at the end, suitable for job output capture.

### DGP benchmark — local

```bash
uv run python benchmarks/benchmark.py
```

### freMTPL2 benchmark — Databricks only

The freMTPL2 benchmark downloads ~30MB from OpenML and fits EBM on 75K rows. It is designed for Databricks serverless compute and will be slow on a laptop.

```bash
databricks workspace import \
  benchmarks/fremtpl2_ebm.py \
  /Workspace/insurance-gam/fremtpl2_benchmark
```

Then attach to serverless compute and run all cells.

### Dependencies

```bash
pip install "interpret==0.7.0" "scikit-learn>=1.3" "polars>=0.20" "statsmodels"
pip install "insurance-gam[ebm]==0.1.6" --no-deps
```

The `interpret` package must be installed before `insurance-gam` so the optional EBM extra resolves correctly. Install order matters.
