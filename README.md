# insurance-gam

**Non-linear tariff models that a pricing actuary can actually read.**

[![PyPI](https://img.shields.io/pypi/v/insurance-gam)](https://pypi.org/project/insurance-gam/) [![Python](https://img.shields.io/pypi/pyversions/insurance-gam)](https://pypi.org/project/insurance-gam/) [![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/burning-cost/insurance-gam/blob/main/LICENSE)

---

## The problem

GLMs need manual feature engineering to capture non-linear effects. A U-shaped driver age curve requires polynomial terms someone has to specify; a convex NCD discount requires a transformation someone has to choose. Get it wrong and the premium is wrong. Get it right and you have a model that looks well-specified but cannot discover interactions you did not anticipate.

GBMs discover those interactions automatically, but the output — thousands of trees — is not auditable by a pricing committee. A pricing actuary cannot look at a gradient booster and tell you whether the NCD discount curve is actuarially reasonable.

GAMs bridge the gap: each feature gets a smooth non-linear shape function, the output is additive and inspectable factor by factor, and interactions can be represented as pairwise 2D shape functions rather than opaque tree splits.

**Blog post:** [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](https://burning-cost.github.io/2026/03/14/insurance-gam-interpretable-nonlinearity/)

---

## Quickstart

```bash
uv add "insurance-gam[ebm]"
```

```python
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(X_train, y_train, exposure=exposure_train)

rt = RelativitiesTable(model)
print(rt.table("driver_age"))   # shape_value, relativity — readable by a pricing actuary
print(rt.summary())
```

Each feature gets a curve. No post-hoc SHAP required — the shape functions are the model.

---

## Validated performance

On a 50,000-policy synthetic UK motor book with a known non-linear DGP (U-shaped driver age, convex NCD, hard vehicle age threshold, log-miles loading):

| Method | Gini vs linear GLM | Poisson deviance |
|---|---|---|
| GLM — linear terms only | baseline | baseline |
| GLM — polynomial + manual interaction | +3–5pp | -2–5% |
| `InsuranceEBM` (interactions=3x) | **+5–15pp** | **-5–12%** |

EBM finds the U-shaped driver age curve and the convex NCD discount without any feature engineering. On a 10,000-policy benchmark, EBM ranks risks ~28% better than a competent GLM by Gini coefficient.

Known caveat: EBM exposure handling via `init_score` can produce inflated absolute deviance figures on some DGPs without affecting risk ordering. Use Gini as the primary comparison metric and validate calibration separately.

Full benchmark: `benchmarks/run_benchmark_databricks.py`.

---

## Why this library?

The PRA expects Pillar 2 capital models to be interpretable. The FCA expects pricing models to be explainable. A black-box GBM satisfies neither requirement for a UK insurer. This library gives you three production-grade GAM variants — EBM, Neural Additive Model, and Pairwise Interaction Networks — that produce per-feature shape functions a pricing actuary can read, challenge, and sign off.

All three use the same GLM-family loss structure (Poisson, Tweedie, Gamma) with exposure offsets, so their outputs are directly comparable to your existing GLM. The subpackages are independent by design: importing `insurance_gam.ebm` does not load PyTorch, and vice versa.

---

## Compared to alternatives

| | Standard GLM | GBM (XGBoost/LightGBM) | R `mgcv` | `interpret.ml` EBM standalone | **insurance-gam** |
|---|---|---|---|---|---|
| Non-linear shape functions | Manual polynomials | Yes (opaque) | Yes | Yes | Yes |
| Per-feature relativity table | Yes (linear) | No | Yes | Partial | Yes (`RelativitiesTable`) |
| Pairwise interactions | Manual dummies | Yes (opaque) | Yes | No | Yes (PIN) |
| Poisson/Gamma/Tweedie loss | Yes | Yes | Yes | No | Yes |
| Exposure offset | Yes | Partial | Yes | No | Yes |
| Python-native | Yes | Yes | No | Yes | Yes |
| PRA/FCA-auditable output | Yes | No | Yes | Partial | Yes |

---

## What's inside

Three subpackages. Import only the one you need.

### `insurance_gam.ebm` — Explainable Boosting Machine

Wraps [interpretML's](https://github.com/interpretml/interpret) `ExplainableBoostingRegressor` with insurance tooling: exposure-aware fit/predict via Poisson/Gamma/Tweedie losses, relativity table extraction, post-fit monotonicity enforcement, and GLM comparison tools.

The `RelativitiesTable` output is directly readable as a rating factor table — NCD years, driver age, vehicle age, each with an auditable curve you can inspect and challenge factor by factor.

```bash
uv add "insurance-gam[ebm]"
```

### `insurance_gam.anam` — Actuarial Neural Additive Model

Neural Additive Model (Laub, Pho, Wong 2025) adapted for insurance. One MLP subnetwork per feature, additive aggregation, Poisson/Tweedie/Gamma losses, Dykstra-projected monotonicity constraints. Beats GLMs on deviance metrics while producing per-feature shape functions a pricing team can inspect.

```bash
uv add "insurance-gam[neural]"
```

```python
from insurance_gam.anam import ANAM

model = ANAM(loss="poisson", monotone_increasing=["vehicle_age"], n_epochs=100)
model.fit(df, y, sample_weight=exposure)
shapes = model.shape_functions()
shapes["vehicle_age"].plot()
```

### `insurance_gam.pin` — Pairwise Interaction Networks

Neural GA2M (Richman, Scognamiglio, Wüthrich 2025). The prediction decomposes as a sum of pairwise interaction terms — one shared network differentiating all feature pairs by learned interaction tokens. Diagonal terms recover main effects. Captures interactions a GLM would miss while keeping the output interpretable as a sum of 2D shape functions.

```bash
uv add "insurance-gam[neural]"
```

```python
from insurance_gam.pin import PINModel

model = PINModel(
    features={"driver_age": "continuous", "vehicle_age": "continuous",
              "area": 5, "ncd_years": "continuous"},
    loss="poisson",
    max_epochs=200,
)
model.fit(df, y, exposure=exposure)
weights = model.interaction_weights()
effects = model.main_effects(df)
```

---

## Installation options

```bash
uv add insurance-gam           # base only (no subpackages loaded)
uv add "insurance-gam[ebm]"    # EBM wrapper (requires interpretML)
uv add "insurance-gam[neural]" # ANAM and PIN (requires PyTorch)
uv add "insurance-gam[all]"    # everything
```

---

## PRA/FCA context

The PRA's Supervisory Statement SS3/18 on model risk management expects firms to demonstrate that models are interpretable and that their outputs can be challenged by subject matter experts. The FCA's Consumer Duty requires pricing models to produce outcomes that can be explained to customers and the regulator.

A GBM satisfies neither criterion for a primary pricing model. The GAM shape functions produced by this library are the actuarial equivalent of the factor curves a pricing committee signs off in a traditional GLM tariff review — except they are fitted automatically rather than hand-crafted.

---

## Design choices

**Three subpackages, independent imports.** Importing `insurance_gam.ebm` does not load PyTorch. Importing `insurance_gam.anam` does not load interpretML. This matters in production where you may have one platform with interpretML but not PyTorch.

**Exposure-aware throughout.** All subpackages accept an `exposure` parameter and use it correctly in the loss function. This is the same GLM family structure pricing teams already use — model outputs are directly comparable to your existing GLM.

**No post-hoc explainability.** The shape functions are the model. You do not need SHAP values to explain why the model charges what it charges.

---

## Limitations

- Below 5,000 policies the EBM boosting procedure can overfit individual bins. Use a GLM below this threshold.
- EBM's `RelativitiesTable` is extracted from additive log-scale contributions, not multiplicative rating factors. The conversion is an approximation when EBM has learnt interaction terms. Cross-validate segment A/E ratios before implementing derived factors in a production tariff.
- `ANAM` and `PINModel` require PyTorch. Fit time on CPU without GPU: 10–30 minutes on complex datasets. EBM fits in 60–120 seconds on a single CPU.
- Monotonicity constraints in `ANAM` use Dykstra projection. Enforcing monotonicity on a factor that genuinely has non-monotone structure (e.g. declaring driver_age monotone when the U-shape is real) will misfit the model.

---

## Part of the Burning Cost stack

Takes smoothed exposure curves from [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) or raw rating factors directly. Feeds fitted tariff models into [insurance-conformal](https://github.com/burning-cost/insurance-conformal), [insurance-fairness](https://github.com/burning-cost/insurance-fairness), and [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). [See the full stack](https://burning-cost.github.io/stack/)

| Library | Description |
|---|---|
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | Rating table smoothing — smoothed Whittaker curves feed into GAM as calibrated inputs |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | FCA proxy discrimination auditing — shape functions make it easier to isolate proxy effects |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — tracks whether GAM shape functions remain calibrated over time |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal inference — establishes whether non-linear effects are genuinely causal |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — uncertainty quantification around GAM predictions |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — sign-off pack for GAM models entering production |

---

## References

**GAM foundations**

- Hastie, T.J. & Tibshirani, R.J. (1990). *Generalized Additive Models*. Chapman & Hall. (Foundational text establishing the backfitting algorithm and GAM theory.)
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press. (Standard reference for mgcv-style penalised regression splines.)

**Explainable Boosting Machines and GA2M**

- Lou, Y., Caruana, R. & Gehrke, J. (2012). "Intelligible models for classification and regression." *KDD 2012*, 150–158. [doi:10.1145/2339530.2339556](https://doi.org/10.1145/2339530.2339556) (Original GA2M paper — pairwise interaction terms in interpretable additive models.)
- Lou, Y., Caruana, R., Gehrke, J. & Hooker, G. (2013). "Accurate intelligible models with pairwise interactions." *KDD 2013*, 623–631. [doi:10.1145/2487575.2487579](https://doi.org/10.1145/2487575.2487579)
- Nori, H., Jenkins, S., Koch, P. & Caruana, R. (2019). "InterpretML: A Unified Framework for Machine Learning Interpretability." [arXiv:1909.09223](https://arxiv.org/abs/1909.09223) (EBM implementation — the software basis for the EBM tariff workflow.)

**Neural Additive Models**

- Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R. & Hinton, G. (2021). "Neural Additive Models: Interpretable Machine Learning with Neural Nets." *NeurIPS 2021*. [arXiv:2004.13912](https://arxiv.org/abs/2004.13912)

**Insurance-specific interpretable modelling**

- Laub, P.J., Pho, K.H. & Wong, T.T. (2025). "An Interpretable Deep Learning Model for General Insurance Pricing." [arXiv:2509.08467](https://arxiv.org/abs/2509.08467)
- Richman, R., Scognamiglio, S. & Wüthrich, M.V. (2025). "Tree-like Pairwise Interaction Networks." [arXiv:2508.15678](https://arxiv.org/abs/2508.15678)
- Denuit, M., Henckaerts, R., Trufin, J. & Verdebout, T. (2021). "Autocalibration and Tweedie-dominance for Insurance Pricing with Machine Learning." *Insurance: Mathematics and Economics*, 101, 485–497. [doi:10.1016/j.insmatheco.2021.09.001](https://doi.org/10.1016/j.insmatheco.2021.09.001)

---

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-gam/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-gam/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 5 covers GAMs and interpretable non-linear models. £97 one-time.

## Licence

MIT
