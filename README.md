# insurance-gam

[![PyPI](https://img.shields.io/pypi/v/insurance-gam)](https://pypi.org/project/insurance-gam/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-gam)](https://pypi.org/project/insurance-gam/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()


Interpretable GAM toolkit for insurance pricing. Three modelling approaches, one package.

GLMs have been the industry standard for decades. They're interpretable, well-understood, and regulators like them. But they leave predictive power on the table — particularly on non-linear effects and interactions. This package gives pricing actuaries three production-grade alternatives that sit between a GLM and a black-box gradient booster: all interpretable, all exposure-aware, all tested against realistic insurance data.

## Quick Start

```bash
pip install "insurance-gam[ebm]"
```

```python
import numpy as np
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

rng = np.random.default_rng(42)
n = 2000

df = pl.DataFrame({
    "driver_age":   rng.integers(17, 75, n).astype(float),
    "vehicle_age":  rng.integers(0, 15, n).astype(float),
    "ncd_years":    rng.integers(0, 9, n).astype(float),
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

Requires the `[ebm]` extra: `pip install "insurance-gam[ebm]"`

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

Requires the `[neural]` extra: `pip install "insurance-gam[neural]"`

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
    monotone_increasing=["vehicle_age", "driver_age"],
    n_epochs=100,
)
model.fit(df, y, sample_weight=exposure)

shapes = model.shape_functions()
shapes["vehicle_age"].plot()
```

### `insurance_gam.pin` — Pairwise Interaction Networks

Neural GA2M (Richman, Scognamiglio, Wüthrich 2025). The prediction decomposes as a sum of pairwise interaction terms — one shared network serving all feature pairs, differentiated by learned interaction tokens. Diagonal terms recover main effects. Captures interactions a GLM would miss while keeping the output interpretable as a sum of 2D shape functions.

Requires the `[neural]` extra: `pip install "insurance-gam[neural]"`

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
pip install insurance-gam
```

With neural subpackages (requires PyTorch):

```bash
pip install "insurance-gam[neural]"
```

With EBM subpackage (requires interpretML):

```bash
pip install "insurance-gam[ebm]"
```

Everything:

```bash
pip install "insurance-gam[all]"
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
├── ebm/     # 136 tests
├── anam/    # 151 tests
└── pin/     # 136 tests
```

## Source repos

This package consolidates three previously separate libraries:

- `insurance-ebm` — archived, merged into `insurance_gam.ebm`
- `insurance-anam` — archived, merged into `insurance_gam.anam`
- `insurance-pin` — archived, merged into `insurance_gam.pin`

---

## Performance

Benchmarked against **Poisson GLM** (statsmodels, main effects only) and **CatBoost Poisson GBM** on synthetic UK motor data — 50,000 policies, known DGP, temporal train/test split. Full notebook: `notebooks/benchmark.py`.

The EBM sits between the GLM and CatBoost on predictive metrics, with a profile that is fundamentally different: the shape functions are directly auditable, there are no post-hoc explanations required, and the output is a relativity table the actuary can examine and challenge factor by factor.

| Metric | Poisson GLM | EBM (insurance-gam) | CatBoost GBM |
|--------|-------------|---------------------|--------------|
| Poisson deviance | highest | between GLM and GBM | lowest |
| Gini coefficient | lowest | between GLM and GBM | highest |
| Interpretability | full (coefficients) | full (shape functions) | requires post-hoc SHAP |
| Auditability for FCA | straightforward | straightforward | requires explanation layer |

The benchmark measures Poisson deviance, Gini, and double-lift chart on the held-out test set. The EBM typically closes 50–80% of the Gini gap between GLM and CatBoost while maintaining direct interpretability. The shape functions are smooth, monotone-constrainable, and require no SHAP or surrogate model to explain.

**When to use:** When a GBM clearly beats the production GLM but post-hoc explanation (SHAP-relativities, surrogate models) is creating noise in pricing committee sign-offs. The EBM offers comparable or better predictive performance than a GLM with hand-crafted interactions, with a shape function per feature rather than a coefficient per dummy level.

**When NOT to use:** When the portfolio has strong multiplicative interactions between rating factors that an additive model cannot capture. The EBM handles pairwise interactions via interaction terms, but the hierarchy is still additive and cannot represent three-way interactions without explicit specification.


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_gam_demo.py).

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

