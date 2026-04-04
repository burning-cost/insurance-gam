# Databricks notebook source
# MAGIC %pip install insurance-gam scikit-learn scipy polars matplotlib -q

# COMMAND ----------

"""
Boulevard EBM Demo — insurance-gam v0.5.1

This notebook demonstrates BoulevardEBM and BoulevardInference on a synthetic
UK motor insurance dataset. The model fits an additive shape function for each
feature and produces CLT-based confidence intervals derived from the known
asymptotic distribution of the Boulevard estimator (Fang et al., AISTATS 2026).

Key results shown:
1. Shape function recovery on a known DGP
2. Confidence interval calibration
3. Significance table — which features drive the model?
4. prediction_interval() for new observations

Boulevard vs. interpretML EBM:
  - No interpretML dependency — lighter install, works everywhere
  - Analytical CIs: no bootstrap needed, O(m³) not O(n·m)
  - MSE loss only in v1 — use InsuranceEBM for Poisson/Tweedie
"""

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from insurance_gam.ebm import BoulevardEBM, BoulevardInference

print("insurance-gam loaded")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 1. Synthetic UK motor data — known additive DGP
# ---------------------------------------------------------------------------
#
# True model (on log scale):
#   y = sin(2*pi * driver_age_norm) + 0.4 * vehicle_age^2 + noise
#
# driver_age_norm: driver_age rescaled to [0,1]
# vehicle_age: 0-15 years, centred

N = 2000
rng = np.random.default_rng(42)

driver_age = rng.uniform(17, 80, N)
vehicle_age = rng.uniform(0, 15, N)
ncd = rng.integers(0, 6, N).astype(float)

# Normalise for shape function
driver_norm = (driver_age - 17) / (80 - 17)
veh_centred = (vehicle_age - 7.5) / 7.5

true_shape_driver = np.sin(2 * np.pi * driver_norm)
true_shape_vehicle = 0.4 * veh_centred ** 2

y = 0.5 + true_shape_driver + true_shape_vehicle + rng.normal(0, 0.3, N)

df = pl.DataFrame({
    "driver_age": driver_age,
    "vehicle_age": vehicle_age,
    "ncd": ncd,
})

print(f"Dataset: {N} rows, {df.shape[1]} features")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 2. Fit BoulevardEBM
# ---------------------------------------------------------------------------

model = BoulevardEBM(
    n_estimators=1000,
    lambda_=1.0,
    max_bins=64,
    min_samples_leaf=20,
    random_state=42,
)
model.fit(df, y)

print(f"Fitted: {model}")
print(f"Intercept: {model.intercept_:.3f}  (true: 0.5)")
print(f"sigma_hat: {model.sigma_hat_:.3f}  (true noise: 0.3)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 3. Wrap with BoulevardInference
# ---------------------------------------------------------------------------

inf = BoulevardInference(model)

# Significance table — quick summary of all features
sig_tbl = inf.significance_table()
print("\nSignificance table:")
print(sig_tbl)

# COMMAND ----------

# ---------------------------------------------------------------------------
# 4. Shape function CIs
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for k, fname in enumerate(model.feature_names_):
    ax = inf.plot_shape(fname, alpha=0.05, ax=axes[k], show_data_density=True)
    ax.set_title(f"Shape: {fname}")

plt.tight_layout()
plt.savefig("/tmp/boulevard_shapes.png", dpi=120, bbox_inches="tight")
print("Shape plots saved to /tmp/boulevard_shapes.png")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 5. Coverage check for driver_age shape
# ---------------------------------------------------------------------------

ci_df = inf.shape_ci("driver_age", alpha=0.05)
print("\ndriver_age CI (first 8 bins):")
print(ci_df.head(8))

# Compare fitted shape vs. true shape at bin midpoints
edges = model.bins_[0]
n_bins = len(model.shape_scores_[0])
mids = np.array([
    (edges[j] + edges[j+1]) / 2 if not np.isinf(edges[j+1]) else edges[j]
    for j in range(n_bins)
])
driver_norm_mids = (mids - 17) / (80 - 17)
true_at_bins = np.sin(2 * np.pi * driver_norm_mids)
true_centred = true_at_bins - true_at_bins.mean()

lower = ci_df["lower"].to_numpy()
upper = ci_df["upper"].to_numpy()
coverage = float(np.mean((lower <= true_centred) & (true_centred <= upper)))
print(f"\n95% CI bin coverage for driver_age: {coverage:.1%}")
assert coverage >= 0.80, f"Coverage too low: {coverage:.1%}"
print("Coverage check passed.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 6. Prediction interval
# ---------------------------------------------------------------------------

pi_df = inf.prediction_interval(df.head(5), alpha=0.05)
print("\nPrediction intervals (first 5 rows):")
print(pi_df)

# Check ordering
assert (pi_df["lower"] <= pi_df["prediction"]).all()
assert (pi_df["prediction"] <= pi_df["upper"]).all()
print("Ordering check passed.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 7. Cross-feature comparison — which features matter?
# ---------------------------------------------------------------------------

print("\nFeature significance summary:")
for row in sig_tbl.iter_rows(named=True):
    sig_marker = "***" if row["ci_excludes_zero"] else ""
    print(
        f"  {row['feature']:15s}  max|score|={row['max_absolute_score']:.3f}  "
        f"max_se={row['max_se']:.3f}  {sig_marker}"
    )

# driver_age and vehicle_age should dominate
driver_row = sig_tbl.filter(pl.col("feature") == "driver_age").row(0, named=True)
ncd_row = sig_tbl.filter(pl.col("feature") == "ncd").row(0, named=True)
assert driver_row["max_absolute_score"] > ncd_row["max_absolute_score"], \
    "driver_age should dominate over ncd"
print("Feature importance ordering check passed.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# 8. Quick R² sanity check
# ---------------------------------------------------------------------------

preds = model.predict(df)
ss_res = float(np.sum((y - preds) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
print(f"\nTraining R²: {r2:.3f}")
assert r2 > 0.4, f"R² too low: {r2:.3f}"
print("R² check passed.")

# COMMAND ----------

print("\nAll Boulevard demo checks passed.")
print("BoulevardEBM and BoulevardInference are production-ready for MSE-loss additive models.")
print("For Poisson/Tweedie, use InsuranceEBM from insurance_gam.ebm.")
