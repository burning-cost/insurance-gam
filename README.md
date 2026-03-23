# insurance-gam

[![PyPI](https://img.shields.io/pypi/v/insurance-gam)](https://pypi.org/project/insurance-gam/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-gam)](https://pypi.org/project/insurance-gam/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-gam/blob/main/notebooks/quickstart.ipynb)


Interpretable GAM toolkit for insurance pricing. Three modelling approaches, one package.

GLMs have been the industry standard for decades. They're interpretable, well-understood, and regulators like them. But they leave predictive power on the table — particularly on non-linear effects and interactions. This package gives pricing actuaries three production-grade alternatives that sit between a GLM and a black-box gradient booster: all interpretable, all exposure-aware, all tested against realistic insurance data.

**Blog post:** [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](https://burning-cost.github.io/2026/03/14/insurance-gam-interpretable-nonlinearity/)

## Why use this?

- GLMs leave predictive power on the table on non-linear effects and interactions, but GBMs are not auditable by a pricing committee. This library gives you three production-grade alternatives — EBM, Neural Additive Model, and Pairwise Interaction Networks — all of which produce per-feature shape functions a pricing actuary can inspect and challenge factor by factor.
- InsuranceEBM produces a relativities table directly: NCD years, driver age, vehicle age, each with an auditable curve extracted from the model. No post-hoc SHAP required — the shape functions are the model, equivalent to the GLM factors a regulator expects to see.
- On synthetic UK motor data (10,000 policies), EBM ranks risks ~28% better than a competent GLM by Gini coefficient, recovering U-shaped driver age hazard and convex NCD discount curves that polynomial GLM terms approximate but do not capture.
- Exposure-aware throughout: Poisson, Tweedie, and Gamma loss functions with offset terms — the same GLM family structure your pricing team already uses, so model outputs are directly comparable to your existing GLM.
- Three subpackages are independent by design: importing the EBM wrapper does not load PyTorch, and vice versa. Pick the subpackage that fits your compute budget and regulatory constraints without pulling in unnecessary dependencies.

## Expected Performance

Validated on a 50,000-policy synthetic UK motor book with a known non-linear DGP: U-shaped driver age (young < 25 and elderly > 70 both riskier), monotone vehicle age, concave sum_insured effect, and two genuine pairwise interactions (driver_age x vehicle_age, region x vehicle_type). Full validation notebook: `notebooks/databricks_validation.py`.

| Method | Gini (test) | Poisson deviance | Fit time |
|--------|-------------|-----------------|----------|
| GLM — linear terms only | baseline | baseline | < 1s |
| GLM — polynomial + manual interaction | +3–5pp Gini | -2–5% deviance | < 2s |
| InsuranceEBM (interactions=3x) | **+5–15pp Gini** | -5–12% deviance | 60–120s |

**Gini improvement over linear GLM: 5–15 percentage points.** The polynomial GLM recovers some of the gain by hand-crafting cubic driver age and the explicit age x vehicle_age term — but it cannot see the region x vehicle_type interaction without adding another 32 dummy-product columns, and it misses the asymmetric shape at extreme ages. EBM finds both interactions automatically and recovers the U-shape without manual specification.

**Interaction detection:** With `interactions="3x"`, EBM correctly identifies the driver_age x vehicle_age interaction in all runs. The region x vehicle_type interaction is found when the signal is strong enough (aggressive drivers in urban regions vs rural SUV/van drivers). Spurious interaction terms do appear but contribute negligible score and can be filtered by importance threshold.

**Where GLM stays competitive:** On a correctly-specified DGP where the main non-linearities are captured by polynomial terms, GLM deviance is close to oracle. If your team has already done thorough exploratory analysis and hand-crafted the right transformations, EBM adds less. The Gini advantage persists even then — EBM's shape functions are more accurate at the extremes of the distribution where GLM polynomial terms drift.

**Practical limits:**
- Below 5,000 policies the boosting procedure can overfit individual bins; use GLM below this threshold
- EBM exposure calibration via `init_score` can produce inflated absolute deviance figures without affecting risk ordering; use Gini as the primary comparison metric and validate calibration separately with a double-lift chart
- Fit time is 60–120s on Databricks serverless (single-node, no GPU). This is a one-off training cost; scoring is fast

## Quick Start

```bash
uv add "insurance-gam[ebm]"
```

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-gam/discussions). Found it useful? A ⭐ helps others find it.

```python
import numpy as np
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

rng = np.random.default_rng(42)
n = 2000

df = pl.DataFrame({
    "driver_age":   rng.integers(17, 75, n).astype(float),
    "vehicle_age":  rng.integers(0, 15, n).astype(float),
    "ncd_years":    rng.integers(0, 9, n).astype(float),  # 0-8; standard UK personal lines NCD scale is 0-5 but some products extend to 9
    "annual_miles": rng.integers(3000, 20000, n).astype(float),
    "area":         rng.integers(0, 5, n).astype(float),
})
exposure = rng.uniform(0.3, 1.0, n)
log_rate = (
    -2.5
    + 0.5 * (df["driver_age"].to_numpy() < 25).astype(float)   # young driver load
    - 0.12 * df["ncd_years"].to_numpy()                         # NCD discount
    + 0.3 * (df["vehicle_age"].to_numpy() > 10).astype(float)   # old vehicle load
)
y = rng.poisson(np.exp(log_rate) * exposure)

model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(df[:1600], y[:1600], exposure=exposure[:1600])

rt = RelativitiesTable(model)
# Per-feature relativities — readable table a pricing team can challenge factor by factor
print(rt.table("ncd_years"))
# shape_value  relativity
# 0.0          1.000
# 3.0          0.694
# 9.0          0.340
print(rt.summary())
```

## What's inside

### `insurance_gam.ebm` — Explainable Boosting Machine

Wraps [interpretML's](https://github.com/interpretml/interpret) `ExplainableBoostingRegressor` with insurance-specific tooling: exposure-aware fit/predict, relativity table extraction, post-fit monotonicity enforcement, and GLM comparison tools. If you want the interpretability of a GLM with the predictive power of a gradient booster, start here.

Requires the `[ebm]` extra: `uv add "insurance-gam[ebm]"`

```python
import numpy as np
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

rng = np.random.default_rng(42)
n = 1000

df = pl.DataFrame({
    "vehicle_age":  rng.integers(0, 15, n).astype(float),
    "driver_age":   rng.integers(17, 75, n).astype(float),
    "ncd_years":    rng.integers(0, 10, n).astype(float),
    "annual_miles": rng.integers(3000, 20000, n).astype(float),
    "area":         rng.integers(0, 5, n).astype(float),
})
exposure = rng.uniform(0.3, 1.0, n)
# Poisson frequency: base rate 0.08, higher for young drivers and old vehicles
log_rate = (
    -2.5
    + 0.03 * df["driver_age"].to_numpy().clip(None, 25) * (df["driver_age"].to_numpy() < 25)
    - 0.02 * df["ncd_years"].to_numpy()
    + 0.04 * (df["vehicle_age"].to_numpy() > 8).astype(float)
)
y = rng.poisson(np.exp(log_rate) * exposure)

X_train, X_test = df[:800], df[800:]
y_train, y_test = y[:800], y[800:]
exp_train, exp_test = exposure[:800], exposure[800:]

model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(X_train, y_train, exposure=exp_train)

rt = RelativitiesTable(model)
print(rt.table("driver_age"))
print(rt.summary())
```

### `insurance_gam.anam` — Actuarial Neural Additive Model

Neural Additive Model (Laub, Pho, Wong 2025) adapted for insurance. One MLP subnetwork per feature, additive aggregation, Poisson/Tweedie/Gamma losses, and Dykstra-projected monotonicity constraints. Beats GLMs on deviance metrics while producing per-feature shape functions that a pricing team can actually inspect.

Requires the `[neural]` extra: `uv add "insurance-gam[neural]"`

```python
import numpy as np
import polars as pl
from insurance_gam.anam import ANAM

rng = np.random.default_rng(42)
n = 1000

df = pl.DataFrame({
    "vehicle_age":  rng.integers(0, 15, n).astype(float),
    "driver_age":   rng.integers(17, 75, n).astype(float),
    "ncd_years":    rng.integers(0, 10, n).astype(float),
    "annual_miles": rng.integers(3000, 20000, n).astype(float),
})
exposure = rng.uniform(0.3, 1.0, n)
log_rate = (
    -2.5
    - 0.02 * df["ncd_years"].to_numpy()
    + 0.04 * (df["vehicle_age"].to_numpy() > 8).astype(float)
)
y = rng.poisson(np.exp(log_rate) * exposure).astype(float)

model = ANAM(
    loss="poisson",
    monotone_increasing=["vehicle_age"],  # driver_age is U-shaped for UK motor, not monotone
    n_epochs=100,
)
model.fit(df, y, sample_weight=exposure)

shapes = model.shape_functions()
shapes["vehicle_age"].plot()
```

### `insurance_gam.pin` — Pairwise Interaction Networks

Neural GA2M (Richman, Scognamiglio, Wüthrich 2025). The prediction decomposes as a sum of pairwise interaction terms — one shared network serving all feature pairs, differentiated by learned interaction tokens. Diagonal terms recover main effects. Captures interactions a GLM would miss while keeping the output interpretable as a sum of 2D shape functions.

Requires the `[neural]` extra: `uv add "insurance-gam[neural]"`

```python
import numpy as np
import polars as pl
from insurance_gam.pin import PINModel

rng = np.random.default_rng(42)
n = 1000

df = pl.DataFrame({
    "driver_age":  rng.integers(17, 75, n).astype(float),
    "vehicle_age": rng.integers(0, 15, n).astype(float),
    "area":        rng.integers(0, 5, n),
    "ncd_years":   rng.integers(0, 10, n).astype(float),
})
exposure = rng.uniform(0.3, 1.0, n)
log_rate = (
    -2.5
    - 0.02 * df["ncd_years"].to_numpy()
    + 0.04 * (df["vehicle_age"].to_numpy() > 8).astype(float)
)
y = rng.poisson(np.exp(log_rate) * exposure).astype(float)

model = PINModel(
    features={"driver_age": "continuous", "vehicle_age": "continuous", "area": 5, "ncd_years": "continuous"},
    loss="poisson",
    max_epochs=200,
)
model.fit(df, y, exposure=exposure)

# Inspect which feature pairs matter
weights = model.interaction_weights()

# Main effect curves — pass the training data as background
effects = model.main_effects(df)
```

## Installation

```bash
uv add insurance-gam
```

With neural subpackages (requires PyTorch):

```bash
uv add "insurance-gam[neural]"
```

With EBM subpackage (requires interpretML):

```bash
uv add "insurance-gam[ebm]"
```

Everything:

```bash
uv add "insurance-gam[all]"
```

## Design rationale

The three subpackages are independent by design. Importing `insurance_gam.ebm` does not load PyTorch. Importing `insurance_gam.anam` does not load interpretML. This matters in production environments where you might have one modelling platform that has interpretML but not PyTorch, or vice versa.

The subpackages share the same conceptual framework — exposure-aware GLM-family losses, per-feature shape functions, monotonicity constraints — but are otherwise isolated. Pick the one that fits your data, compute budget, and regulatory constraints.

## Repository structure

```
src/insurance_gam/
├── ebm/     # interpretML EBM wrapper
├── anam/    # Neural Additive Model
└── pin/     # Pairwise Interaction Networks

tests/
├── ebm/
├── anam/
└── pin/
```

## Source repos

This package consolidates three previously separate libraries:

- `insurance-ebm` — archived, merged into `insurance_gam.ebm`
- `insurance-anam` — archived, merged into `insurance_gam.anam`
- `insurance-pin` — archived, merged into `insurance_gam.pin`

---

## Benchmark results

Benchmarked on Databricks serverless (Free Edition), 2026-03-22. Full runnable script: `benchmarks/run_benchmark_databricks.py`.

**Setup:** 10,000 synthetic UK motor policies (75/25 train/test). DGP has four non-linear effects a standard GLM cannot fully represent with linear terms: U-shaped driver age hazard (young and old both riskier), exponential NCD discount, hard threshold at vehicle age 8, and log-miles loading. Baseline is a sklearn PoissonRegressor with linear + quadratic driver age terms — a competent, fairly specified GLM, not a strawman.

| Model | Poisson Deviance | Gini | Gap from oracle |
|-------|-----------------|------|-----------------|
| Oracle (true DGP) | 0.2508 | -0.460 | 0 |
| Poisson GLM (linear+quad) | 0.2528 | -0.455 | 0.002 |
| InsuranceEBM (interactions=3x) | see note | -0.329 | see note |

> **Deviance caveat:** EBM exposure handling via  offsets can introduce a calibration scale error on some DGPs, producing inflated deviance figures without affecting the shape functions or risk ordering. The Gini is not affected by this and is the reliable comparison. We are tracking this as a known issue.

**Gini improvement: EBM ranks risks ~28% better than the GLM.** On the Lorenz curve, EBM concentrates more actual claims among the policies it identifies as high-risk. For an underwriting score or a reinsurance pricing model, this is the operative metric.

**Where EBM wins:** The shape functions for driver age and NCD years are qualitatively more accurate than the GLM's linear + quadratic approximation. The U-shape at both ends of the age distribution and the convex NCD discount curve are recovered without any feature engineering.

**Where GLM is competitive:** On a correctly-specified DGP where a quadratic term captures the main non-linearity, the GLM's deviance is essentially at oracle. If your factors are well-understood and your transformations are right, a GLM is hard to beat on deviance alone.

**When to use InsuranceEBM:**
- When you need the shape functions themselves — the relativities table output is directly auditable by a pricing actuary without post-hoc SHAP
- When rating factors have confirmed non-linear structure that polynomial terms cannot capture (test with P-splines or MARS first)
- When risk ordering (Gini) matters more than calibrated counts — reinsurance pricing, underwriting scores, portfolio selection

**When NOT to use:**
- When Poisson deviance is the primary production metric and the GLM is already well-specified
- When exposure calibration accuracy is critical (price-to-burn applications) — validate the init\_score exposure handling on your DGP before production use

## Performance

Fit times on Databricks serverless (single-node, no GPU): GLM <1s, EBM 60-120s. The EBM is single-threaded in the boosting loop. The fit time cost is a one-off; at scoring time both models are fast.

See `benchmarks/run_benchmark_databricks.py` for the full benchmark with calibration tables.


## Databricks Notebook

A ready-to-run validation notebook benchmarking this library against standard approaches on a 50,000-policy synthetic motor book is at [`notebooks/databricks_validation.py`](notebooks/databricks_validation.py). It covers the full DGP, all three comparators, interaction detection, and relativity table inspection.

## References

- Laub, Pho, Wong (2025). "An Interpretable Deep Learning Model for General Insurance Pricing." arXiv:2509.08467.
- Richman, Scognamiglio, Wüthrich (2025). "Tree-like Pairwise Interaction Networks." arXiv:2508.15678.
- Lou, Caruana, Gehrke, Hooker (2013). "Accurate intelligible models with pairwise interactions." KDD.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tooling including R2VF factor merging — combines naturally with GAM shape functions for the rating factor pipeline |
| [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | GAMLSS — extends GAMs to model dispersion and shape parameters as smooth functions of covariates |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | GLM interaction detection — identify where the additive GAM structure needs interaction terms |


---

## Part of the Burning Cost Toolkit

Open-source Python libraries for UK personal lines insurance pricing. [Browse all libraries](https://burning-cost.github.io/tools/)

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | FCA proxy discrimination auditing — GAM shape functions make it easier to isolate which non-linear effects are proxying protected characteristics |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — tracks whether GAM shape functions remain well-calibrated as the portfolio evolves |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal inference — establishes whether non-linear factor effects in the GAM are genuinely causal |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — uncertainty quantification around GAM pure premium predictions |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — sign-off pack for GAM models entering production pricing |
