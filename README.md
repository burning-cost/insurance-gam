# insurance-gam

Interpretable GAM toolkit for insurance pricing. Three modelling approaches, one package.

GLMs have been the industry standard for decades. They're interpretable, well-understood, and regulators like them. But they leave predictive power on the table — particularly on non-linear effects and interactions. This package gives pricing actuaries three production-grade alternatives that sit between a GLM and a black-box gradient booster: all interpretable, all exposure-aware, all tested against realistic insurance data.

## What's inside

### `insurance_gam.ebm` — Explainable Boosting Machine

Wraps [interpretML's](https://github.com/interpretml/interpret) `ExplainableBoostingRegressor` with insurance-specific tooling: exposure-aware fit/predict, relativity table extraction, post-fit monotonicity enforcement, and GLM comparison tools. If you want the interpretability of a GLM with the predictive power of a gradient booster, start here.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(X_train, y_train, exposure=exposure_train)

rt = RelativitiesTable(model)
print(rt.table("driver_age"))
print(rt.summary())
```

### `insurance_gam.anam` — Actuarial Neural Additive Model

Neural Additive Model (Laub, Pho, Wong 2025) adapted for insurance. One MLP subnetwork per feature, additive aggregation, Poisson/Tweedie/Gamma losses, and Dykstra-projected monotonicity constraints. Beats GLMs on deviance metrics while producing per-feature shape functions that a pricing team can actually inspect.

```python
from insurance_gam.anam import ANAM

model = ANAM(
    loss="poisson",
    monotone_increasing=["vehicle_age", "driver_age"],
    n_epochs=100,
)
model.fit(X_train, y_train, sample_weight=exposure_train)

shapes = model.shape_functions()
shapes["vehicle_age"].plot()
```

### `insurance_gam.pin` — Pairwise Interaction Networks

Neural GA2M (Richman, Scognamiglio, Wüthrich 2025). The prediction decomposes as a sum of pairwise interaction terms — one shared network serving all feature pairs, differentiated by learned interaction tokens. Diagonal terms recover main effects. Captures interactions a GLM would miss while keeping the output interpretable as a sum of 2D shape functions.

```python
from insurance_gam.pin import PINModel

model = PINModel(
    features={"driver_age": "continuous", "vehicle_age": "continuous", "area": 5},
    loss="poisson",
    max_epochs=200,
)
model.fit(X_train, y_train, exposure=exposure_train)

# Inspect which feature pairs matter
weights = model.interaction_weights()
effects = model.main_effects(X_background)
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

## References

- Laub, Pho, Wong (2025). "An Interpretable Deep Learning Model for General Insurance Pricing." arXiv:2509.08467.
- Richman, Scognamiglio, Wüthrich (2025). "Tree-like Pairwise Interaction Networks." arXiv:2508.15678.
- Lou, Caruana, Gehrke, Hooker (2013). "Accurate intelligible models with pairwise interactions." KDD.
