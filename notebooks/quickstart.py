# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-gam: Interpretable GAM Toolkit for Insurance Pricing
# MAGIC
# MAGIC GLMs are the industry standard in UK insurance pricing: interpretable, well-understood, regulator-friendly. But they leave predictive power on the table when non-linear effects and interactions matter.
# MAGIC
# MAGIC This package gives you three alternatives that sit between a GLM and a black-box gradient booster: all interpretable, all exposure-aware, all testable. This notebook covers the most accessible entry point: `InsuranceEBM` (Explainable Boosting Machine) from the `ebm` subpackage.

# COMMAND ----------

# MAGIC %pip install "insurance-gam[ebm]" polars numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Synthetic Motor Pricing Data
# MAGIC
# MAGIC 3,000 policies with a Poisson claim count target. The true model has:
# MAGIC - A non-linear young-driver effect (age < 25 adds 50% frequency)
# MAGIC - A linear NCD discount (−12% per year)
# MAGIC - An old-vehicle load (vehicle age > 10 adds 30%)
# MAGIC
# MAGIC No interaction between rating factors. The EBM should recover this cleanly.

# COMMAND ----------

rng = np.random.default_rng(42)
n = 3_000

df = pl.DataFrame({
    "driver_age":    rng.integers(17, 75, n).astype(float),
    "vehicle_age":   rng.integers(0, 15, n).astype(float),
    "ncd_years":     rng.integers(0, 9, n).astype(float),
    "annual_miles":  rng.integers(3000, 20000, n).astype(float),
    "area":          rng.integers(0, 5, n).astype(float),
})
exposure = rng.uniform(0.3, 1.0, n)

log_rate = (
    -2.5
    + 0.5 * (df["driver_age"].to_numpy() < 25).astype(float)
    - 0.12 * df["ncd_years"].to_numpy()
    + 0.3 * (df["vehicle_age"].to_numpy() > 10).astype(float)
)
y = rng.poisson(np.exp(log_rate) * exposure)

print(f"n={n:,} policies | mean claim freq: {(y/exposure).mean():.4f}")
print(f"Claims: {y.sum()} total")
df.head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit the EBM
# MAGIC
# MAGIC `InsuranceEBM` wraps interpretML's ExplainableBoostingRegressor with insurance-specific defaults: Poisson loss, exposure offset, and a `RelativitiesTable` that produces the kind of output a pricing actuary can review factor by factor.

# COMMAND ----------

from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

train_n = 2400
model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(df[:train_n], y[:train_n], exposure=exposure[:train_n])

print("EBM fitted. Feature importances:")
importances = model.feature_importances()
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = '#' * int(imp * 40)
    print(f"  {feat:<20} {imp:.4f}  {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Relativities Table
# MAGIC
# MAGIC `RelativitiesTable` converts EBM shape functions into the multiplicative relativity tables pricing teams use. Each value is relative to the grand mean — 1.0 means no loading, 1.5 means +50%.
# MAGIC
# MAGIC This is the output format a pricing actuary can compare against their existing GLM relativities and challenge rationally.

# COMMAND ----------

rt = RelativitiesTable(model)

print("NCD years relativities:")
print(rt.table("ncd_years").to_pandas().to_string(index=False))

print()
print("Driver age relativities (first 8 bands):")
age_rel = rt.table("driver_age")
print(age_rel.head(8).to_pandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Out-of-Sample Gini
# MAGIC
# MAGIC The acid test: does the EBM discriminate better than a flat GLM on held-out data?

# COMMAND ----------

from insurance_gam.ebm import gini

y_test = y[train_n:]
exposure_test = exposure[train_n:]
y_pred_ebm = model.predict(df[train_n:])
y_pred_flat = np.full_like(y_pred_ebm, y_pred_ebm.mean())

gini_ebm  = gini(y_test, y_pred_ebm,  exposure=exposure_test)
gini_flat = gini(y_test, y_pred_flat, exposure=exposure_test)

print(f"Gini (EBM):       {gini_ebm:.3f}")
print(f"Gini (flat null): {gini_flat:.3f}")
print(f"Lift over null:   {gini_ebm - gini_flat:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What You Should See
# MAGIC
# MAGIC - `ncd_years` and `driver_age` should be the top two features by importance.
# MAGIC - NCD relativities should be monotone decreasing (0 NCD = highest multiplier, 8 NCD = lowest).
# MAGIC - EBM Gini should be materially above the flat model Gini.
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC - **`MonotonicityEditor`** — enforce monotone constraints on any feature (NCD must decrease, age must be U-shaped)
# MAGIC - **`GLMComparison`** — run a side-by-side lift comparison between the EBM and an existing GLM on the same holdout
# MAGIC - **`insurance_gam.anam`** — Actuarial Neural Additive Model (deep learning, same interpretability)
# MAGIC - **`insurance_gam.pin`** — Pairwise Interaction Networks for modelling two-way interactions explicitly
# MAGIC
# MAGIC **GitHub:** https://github.com/burning-cost/insurance-gam
# MAGIC **PyPI:** https://pypi.org/project/insurance-gam/
