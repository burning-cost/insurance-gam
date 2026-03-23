# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-gam: Validation on a Synthetic UK Motor Portfolio
# MAGIC
# MAGIC This notebook validates insurance-gam on a realistic synthetic motor portfolio.
# MAGIC
# MAGIC The central claim of this library is that a standard Poisson GLM leaves predictive
# MAGIC power on the table when rating factors have non-linear shapes and genuine pairwise
# MAGIC interactions — both common in UK motor data. InsuranceEBM (Explainable Boosting Machine)
# MAGIC captures these shapes automatically without manual feature engineering, while remaining
# MAGIC fully interpretable through per-feature relativity tables.
# MAGIC
# MAGIC What this notebook shows:
# MAGIC
# MAGIC 1. A 50,000-policy synthetic UK motor book with a known non-linear DGP
# MAGIC    (U-shaped driver age, monotone vehicle age, concave sum_insured) and 2 genuine
# MAGIC    pairwise interactions (driver_age x vehicle_age, region x vehicle_type)
# MAGIC 2. Standard Poisson GLM with linear terms — what most teams currently do
# MAGIC 3. GLM with manual polynomial terms — a competent, well-specified baseline
# MAGIC 4. InsuranceEBM — what this library does
# MAGIC 5. Comparison table: deviance, Gini, interaction detection accuracy
# MAGIC 6. Relativity table inspection: does EBM recover the true factor shapes?
# MAGIC
# MAGIC **Expected result:** EBM improves Gini by 5-15pp over the linear GLM and 3-8pp over
# MAGIC the polynomial GLM, while correctly identifying the genuine interactions from noise.
# MAGIC Shapley-equivalent relativities are directly readable by a pricing actuary without
# MAGIC any post-hoc explanation.
# MAGIC
# MAGIC ---
# MAGIC *Part of the [Burning Cost](https://burning-cost.github.io) insurance pricing toolkit.*

# COMMAND ----------

# MAGIC %pip install "insurance-gam[ebm]" polars scikit-learn -q

# COMMAND ----------

from __future__ import annotations

import time
import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data-Generating Process
# MAGIC
# MAGIC The DGP is a 50,000-policy UK motor frequency book. The key structural features are:
# MAGIC
# MAGIC - **driver_age**: U-shaped hazard — young drivers (<25) and elderly drivers (>70) are both
# MAGIC   riskier than the middle band. A linear term cannot capture this; a quadratic approximates
# MAGIC   it but misses the asymmetry.
# MAGIC - **vehicle_age**: monotone increasing — older vehicles have higher claim frequency (worse
# MAGIC   safety systems, more mechanical failures). A linear term works here, but EBM will verify it.
# MAGIC - **sum_insured** (as proxy for vehicle value): concave — very cheap and very expensive
# MAGIC   vehicles are both higher risk than mid-range. Again, non-linear structure.
# MAGIC - **Interaction 1**: driver_age x vehicle_age — young drivers in old vehicles are
# MAGIC   disproportionately risky beyond the main effects. A GLM without explicit interaction
# MAGIC   terms cannot see this.
# MAGIC - **Interaction 2**: region x vehicle_type — urban regions see higher frequency on
# MAGIC   hatchbacks/saloons; rural regions see higher frequency on SUVs/vans (different driving
# MAGIC   patterns). GLM with linear main effects misses this.
# MAGIC
# MAGIC The true DGP is a Poisson frequency model. All three GLM approaches and the EBM use
# MAGIC the same Poisson loss, so the comparison is methodologically clean.

# COMMAND ----------

RNG = np.random.default_rng(42)
N = 50_000
N_TRAIN = 40_000
N_REGIONS = 8
N_VEHICLE_TYPES = 4

# --- Rating factors ---
driver_age    = RNG.integers(17, 80, N).astype(float)
vehicle_age   = RNG.integers(0, 20, N).astype(float)
sum_insured   = RNG.uniform(4_000, 60_000, N)  # vehicle value in GBP
ncb_years     = RNG.integers(0, 9, N).astype(float)
region        = RNG.integers(0, N_REGIONS, N)
vehicle_type  = RNG.integers(0, N_VEHICLE_TYPES, N)  # 0=hatchback, 1=saloon, 2=suv, 3=van
exposure      = RNG.uniform(0.3, 1.0, N)

# --- True non-linear main effects (log scale) ---

# U-shaped driver age: base at age 40, increased risk at young and old extremes
# True shape: -0.03*(age-40)^2/100 + 0.8*(age<25) + 0.4*(age>70)
# This is genuinely non-quadratic because the young/old asymmetry differs
age_effect = (
    -0.0003 * (driver_age - 40.0) ** 2
    + 0.9 * (driver_age < 22).astype(float)
    + 0.5 * (driver_age < 25).astype(float)
    + 0.35 * (driver_age > 70).astype(float)
    + 0.6 * (driver_age > 75).astype(float)
)

# Vehicle age: monotone increasing, steeper for very old vehicles
veh_age_effect = 0.035 * vehicle_age + 0.018 * np.maximum(vehicle_age - 12, 0)

# NCD: exponential discount (concave — steeper at low NCD years)
ncd_effect = -0.15 * ncb_years + 0.04 * ncb_years ** 2 * (ncb_years < 5).astype(float) * (-1)

# Sum insured: concave (cheap and expensive vehicles both higher risk)
# Normalise to [0,1] for tractability
si_norm = (sum_insured - 4_000) / 56_000
si_effect = -2.0 * (si_norm - 0.3) ** 2 + 0.3 * (si_norm < 0.1).astype(float)

# Region: random fixed effects
region_effects = np.array([-0.15, 0.20, -0.05, 0.30, 0.10, -0.20, 0.05, 0.15])
region_effect = region_effects[region]

# Vehicle type: random fixed effects
vtype_effects = np.array([0.05, 0.0, 0.10, 0.20])
vtype_effect = vtype_effects[vehicle_type]

# --- Interactions ---
# Interaction 1: young driver (age < 25) x old vehicle (age > 10) — additional load
young_old_interaction = (
    0.45
    * (driver_age < 25).astype(float)
    * (vehicle_age > 10).astype(float)
)

# Interaction 2: region x vehicle_type — urban regions (0,1,3) x hatchback/saloon load
urban_regions = np.isin(region, [0, 1, 3]).astype(float)
hatchback_saloon = (vehicle_type <= 1).astype(float)
rural_suv_van = np.isin(region, [2, 4, 5, 6, 7]).astype(float) * (vehicle_type >= 2).astype(float)

region_vtype_interaction = (
    0.25 * urban_regions * hatchback_saloon
    + 0.20 * rural_suv_van
)

# --- Compose log rate ---
TRUE_BASE_RATE = -2.8  # ~6% annual claim frequency
log_rate = (
    TRUE_BASE_RATE
    + age_effect
    + veh_age_effect
    + ncd_effect
    + si_effect
    + region_effect
    + vtype_effect
    + young_old_interaction
    + region_vtype_interaction
    + RNG.normal(0, 0.05, N)   # small unexplained noise
)

# Poisson claim counts with exposure offset
lambda_i = np.exp(log_rate) * exposure
y = RNG.poisson(lambda_i).astype(float)

# Train/test split
train = np.arange(N_TRAIN)
test  = np.arange(N_TRAIN, N)

print(f"Portfolio: {N:,} policies  ({N_TRAIN:,} train / {N - N_TRAIN:,} test)")
print(f"Mean claim frequency (train): {(y[train] / exposure[train]).mean():.4f}")
print(f"Mean claim frequency (test):  {(y[test]  / exposure[test]).mean():.4f}")
print(f"Overall claim frequency:      {y.mean() / exposure.mean():.4f}")
print()
print("Factor ranges (train):")
print(f"  driver_age:   [{driver_age[train].min():.0f}, {driver_age[train].max():.0f}]")
print(f"  vehicle_age:  [{vehicle_age[train].min():.0f}, {vehicle_age[train].max():.0f}]")
print(f"  sum_insured:  [{sum_insured[train].min():.0f}, {sum_insured[train].max():.0f}]")
print(f"  ncb_years:    [{ncb_years[train].min():.0f}, {ncb_years[train].max():.0f}]")
print()
print("DGP interactions:")
print(f"  Young (<25) x Old vehicle (>10): {young_old_interaction[train].mean():.3f} avg log-rate load")
print(f"  Region x vehicle_type:           {region_vtype_interaction[train].mean():.3f} avg log-rate load")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Estimator (a) — Standard Poisson GLM with Linear Terms
# MAGIC
# MAGIC This is what most UK pricing teams produce as their starting point. Linear main effects
# MAGIC only, no polynomial terms, no interactions. Fast to fit, easy to explain, but it cannot
# MAGIC capture the U-shaped driver age effect, the concave sum_insured effect, or either interaction.
# MAGIC
# MAGIC Exposure is handled as an offset (log(exposure) added to the linear predictor) — the
# MAGIC standard GLM treatment.

# COMMAND ----------

from insurance_gam.ebm import gini, deviance as ebm_deviance

# Build feature matrices — convert region and vehicle_type to dummies
def make_glm_features(driver_age_, vehicle_age_, ncb_, si_, region_, vtype_, include_polys=False):
    """Build feature matrix for GLM. include_polys adds polynomial and interaction terms."""
    feats = {
        "driver_age": driver_age_,
        "vehicle_age": vehicle_age_,
        "ncb_years": ncb_,
        "sum_insured": si_,
    }
    if include_polys:
        feats["driver_age_sq"] = driver_age_ ** 2
        feats["driver_age_cu"] = driver_age_ ** 3
        feats["vehicle_age_sq"] = vehicle_age_ ** 2
        feats["si_sq"] = si_ ** 2
        feats["age_x_vehage"] = driver_age_ * vehicle_age_ / 1000.0
    # Region and vehicle_type dummies (drop first category)
    for r in range(1, N_REGIONS):
        feats[f"region_{r}"] = (region_ == r).astype(float)
    for v in range(1, N_VEHICLE_TYPES):
        feats[f"vtype_{v}"] = (vtype_ == v).astype(float)
    return np.column_stack(list(feats.values()))

X_tr_lin = make_glm_features(
    driver_age[train], vehicle_age[train], ncb_years[train],
    sum_insured[train], region[train], vehicle_type[train]
)
X_te_lin = make_glm_features(
    driver_age[test], vehicle_age[test], ncb_years[test],
    sum_insured[test], region[test], vehicle_type[test]
)

t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_linear = PoissonRegressor(alpha=0, max_iter=500)
    glm_linear.fit(X_tr_lin, y[train] / exposure[train], sample_weight=exposure[train])
t_glm_lin = time.perf_counter() - t0

pred_linear = glm_linear.predict(X_te_lin) * exposure[test]
gini_linear = gini(y[test], pred_linear, exposure[test])
dev_linear  = ebm_deviance(y[test], pred_linear, exposure[test], family="poisson")

print(f"GLM (linear)  |  Fit time: {t_glm_lin:.2f}s")
print(f"  Poisson deviance (test): {dev_linear:.5f}")
print(f"  Gini coefficient (test): {gini_linear:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estimator (b) — Poisson GLM with Polynomial Terms
# MAGIC
# MAGIC A competent, well-specified GLM: cubic driver age, quadratic vehicle age, quadratic
# MAGIC sum_insured, and an explicit driver_age x vehicle_age interaction term. This is what
# MAGIC a skilled pricing actuary would build after exploratory analysis.
# MAGIC
# MAGIC It still cannot capture the region x vehicle_type interaction without adding another 32
# MAGIC dummy-product columns. The U-shape at extreme ages is approximated but not exact.

# COMMAND ----------

X_tr_poly = make_glm_features(
    driver_age[train], vehicle_age[train], ncb_years[train],
    sum_insured[train], region[train], vehicle_type[train],
    include_polys=True
)
X_te_poly = make_glm_features(
    driver_age[test], vehicle_age[test], ncb_years[test],
    sum_insured[test], region[test], vehicle_type[test],
    include_polys=True
)

t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_poly = PoissonRegressor(alpha=1e-4, max_iter=1000)
    glm_poly.fit(X_tr_poly, y[train] / exposure[train], sample_weight=exposure[train])
t_glm_poly = time.perf_counter() - t0

pred_poly = glm_poly.predict(X_te_poly) * exposure[test]
gini_poly = gini(y[test], pred_poly, exposure[test])
dev_poly  = ebm_deviance(y[test], pred_poly, exposure[test], family="poisson")

print(f"GLM (polynomial) | Fit time: {t_glm_poly:.2f}s")
print(f"  Poisson deviance (test): {dev_poly:.5f}")
print(f"  Gini coefficient (test): {gini_poly:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Estimator (c) — InsuranceEBM
# MAGIC
# MAGIC The EBM fits piecewise-smooth shape functions for each feature plus pairwise interaction
# MAGIC terms. It learns the U-shape in driver age, the concavity in sum_insured, and the
# MAGIC interactions without being told what to look for.
# MAGIC
# MAGIC Key design choices for insurance:
# MAGIC - `loss="poisson"` — Poisson deviance loss matches the GLM family
# MAGIC - `interactions="3x"` — up to 3 x n_features interaction terms; EBM selects the most
# MAGIC   important ones automatically via residual fitting
# MAGIC - Exposure via `init_score=log(exposure)` — same offset approach as a GLM
# MAGIC
# MAGIC This step takes 60-120 seconds on Databricks serverless. The fit is single-threaded in
# MAGIC the boosting loop (interpretML design). The cost is one-off at training time.

# COMMAND ----------

from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

df_train = pl.DataFrame({
    "driver_age":    driver_age[train],
    "vehicle_age":   vehicle_age[train],
    "ncb_years":     ncb_years[train],
    "sum_insured":   sum_insured[train],
    "region":        region[train].astype(float),
    "vehicle_type":  vehicle_type[train].astype(float),
})

df_test = pl.DataFrame({
    "driver_age":    driver_age[test],
    "vehicle_age":   vehicle_age[test],
    "ncb_years":     ncb_years[test],
    "sum_insured":   sum_insured[test],
    "region":        region[test].astype(float),
    "vehicle_type":  vehicle_type[test].astype(float),
})

t0 = time.perf_counter()
ebm = InsuranceEBM(loss="poisson", interactions="3x", random_state=42)
ebm.fit(df_train, y[train], exposure=exposure[train])
t_ebm = time.perf_counter() - t0

pred_ebm = ebm.predict(df_test, exposure=exposure[test])
gini_ebm  = gini(y[test], pred_ebm, exposure[test])
dev_ebm   = ebm_deviance(y[test], pred_ebm, exposure[test], family="poisson")

print(f"InsuranceEBM | Fit time: {t_ebm:.1f}s")
print(f"  Poisson deviance (test): {dev_ebm:.5f}")
print(f"  Gini coefficient (test): {gini_ebm:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Interaction Detection
# MAGIC
# MAGIC The EBM's interaction selection acts as an implicit statistical test: if EBM puts an
# MAGIC interaction term in the model, it found enough signal there to improve the deviance.
# MAGIC We check whether the two genuine interactions from the DGP (driver_age x vehicle_age,
# MAGIC region x vehicle_type) appear in the top interaction terms, versus any spurious ones.

# COMMAND ----------

# Extract interaction terms from the fitted EBM
# interpretML stores interaction names as tuples in feature_names_in_
try:
    all_names = list(ebm.ebm_.feature_names_in_)
    interaction_terms = [n for n in all_names if " x " in str(n) or "&" in str(n) or (isinstance(n, tuple))]
    # Also check string representation
    interaction_names_str = [str(n) for n in all_names if isinstance(n, (list, tuple))]

    # Get global explanation to see interaction strengths
    explanation = ebm.ebm_.explain_global(name="EBM")

    print("Feature names in fitted EBM:")
    for i, name in enumerate(all_names):
        if isinstance(name, (list, tuple)) or " & " in str(name):
            print(f"  [interaction] {name}")

    # Check for presence of genuine interactions
    names_str = [str(n).lower() for n in all_names]
    genuine_1_found = any(
        ("driver_age" in n and "vehicle_age" in n) for n in names_str
    )
    genuine_2_found = any(
        ("region" in n and "vehicle_type" in n) for n in names_str
    )
    n_interaction_terms = sum(isinstance(n, (list, tuple)) or " & " in str(n) for n in all_names)

    print()
    print(f"Interaction terms in model: {n_interaction_terms}")
    print(f"Genuine interaction 1 (driver_age x vehicle_age) detected: {genuine_1_found}")
    print(f"Genuine interaction 2 (region x vehicle_type) detected:    {genuine_2_found}")
except Exception as e:
    print(f"Interaction inspection note: {e}")
    print("(EBM fitted — feature names accessible via ebm.ebm_.feature_names_in_)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Relativity Table Inspection
# MAGIC
# MAGIC The relativity table is the main interpretability output. It shows the fitted shape
# MAGIC function for each feature as a table of bins and multiplicative relativities — the
# MAGIC same format a pricing actuary would see from a GLM factor table, but derived from
# MAGIC the EBM's non-parametric shape without manual specification.
# MAGIC
# MAGIC Key checks:
# MAGIC - driver_age: does it recover the U-shape (relativities > 1.0 at low and high ages,
# MAGIC   minimum around 35-50)?
# MAGIC - vehicle_age: does it recover the monotone increasing pattern?
# MAGIC - sum_insured: does it recover the concave shape (higher relativities at extreme values)?

# COMMAND ----------

rt = RelativitiesTable(ebm)

print("=== DRIVER AGE relativities (first 15 bins, then last 5) ===")
age_table = rt.table("driver_age")
print("First 15 bins:")
print(age_table.head(15).to_pandas().to_string(index=False))
print("Last 5 bins:")
print(age_table.tail(5).to_pandas().to_string(index=False))

# Check U-shape: young bins (age<25) and old bins (age>70) should have higher relativities
# than middle bins (age 35-55)
age_table_np = age_table.to_pandas()
# Identify young bins (labels containing values < 25)
# We can check by bin index (first few vs last few vs middle)
n_bins = len(age_table_np)
young_idx = list(range(0, max(3, n_bins // 8)))
middle_idx = list(range(n_bins // 4, 3 * n_bins // 4))
old_idx = list(range(max(0, n_bins - n_bins // 8), n_bins))

young_mean_rel = age_table_np["relativity"].iloc[young_idx].mean()
middle_mean_rel = age_table_np["relativity"].iloc[middle_idx].mean()
old_mean_rel = age_table_np["relativity"].iloc[old_idx].mean()

print(f"\nU-shape check:")
print(f"  Mean relativity — young age bins (first {len(young_idx)}/{n_bins}):  {young_mean_rel:.3f}")
print(f"  Mean relativity — middle age bins ({len(middle_idx)}/{n_bins}):      {middle_mean_rel:.3f}")
print(f"  Mean relativity — old age bins (last {len(old_idx)}/{n_bins}):       {old_mean_rel:.3f}")
u_shape_detected = (young_mean_rel > middle_mean_rel) and (old_mean_rel > middle_mean_rel)
print(f"  U-shape recovered: {u_shape_detected}")

# COMMAND ----------

print("=== VEHICLE AGE relativities ===")
veh_table = rt.table("vehicle_age")
print(veh_table.to_pandas().to_string(index=False))
veh_rels = veh_table["relativity"].to_numpy()
monotone_check = all(veh_rels[i] <= veh_rels[i+1] + 0.05 for i in range(len(veh_rels)-1))
print(f"\nMonotone increasing check (tolerance 0.05): {monotone_check}")

# COMMAND ----------

print("=== RELATIVITY TABLE SUMMARY (all features) ===")
summary = rt.summary()
print(summary.to_pandas().to_string(index=False))
print()
print("The 'range' column shows max/min relativity — a measure of each factor's leverage.")
print("Driver age and vehicle age should have the widest range on this DGP.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Results Summary

# COMMAND ----------

print("=" * 75)
print("VALIDATION SUMMARY — 50,000-policy UK motor, non-linear DGP + 2 interactions")
print("=" * 75)
print(f"{'Method':<30} {'Gini':>10} {'Poisson Dev':>14} {'Fit time':>12}")
print("-" * 75)
print(f"{'GLM (linear terms)':<30} {gini_linear:>10.4f} {dev_linear:>14.5f} {t_glm_lin:>11.2f}s")
print(f"{'GLM (poly + age*veh interaction)':<30} {gini_poly:>10.4f} {dev_poly:>14.5f} {t_glm_poly:>11.2f}s")
print(f"{'InsuranceEBM (interactions=3x)':<30} {gini_ebm:>10.4f} {dev_ebm:>14.5f} {t_ebm:>11.1f}s")
print()

gini_improvement_vs_linear = (gini_ebm - gini_linear) * 100
gini_improvement_vs_poly   = (gini_ebm - gini_poly)   * 100
dev_improvement_vs_linear  = (dev_linear - dev_ebm) / dev_linear * 100
dev_improvement_vs_poly    = (dev_poly - dev_ebm)   / dev_poly   * 100

print(f"Gini improvement (EBM vs linear GLM):     +{gini_improvement_vs_linear:.1f}pp")
print(f"Gini improvement (EBM vs polynomial GLM): +{gini_improvement_vs_poly:.1f}pp")
print(f"Deviance improvement (EBM vs linear GLM): {dev_improvement_vs_linear:+.1f}%")
print(f"Deviance improvement (EBM vs polynomial): {dev_improvement_vs_poly:+.1f}%")
print()
print("EXPECTED PERFORMANCE (50k-policy motor book, non-linear DGP):")
print("  EBM Gini improvement vs linear GLM:      5-15pp")
print("  EBM Gini improvement vs polynomial GLM:  3-8pp")
print("  Interaction detection (top-k accuracy):  >=1 of 2 genuine interactions found")
print("  U-shape in driver_age recovered:         expected yes")
print("  Vehicle_age monotonicity:                expected yes")
print()
print(f"Interaction detection:")
try:
    print(f"  driver_age x vehicle_age found: {genuine_1_found}")
    print(f"  region x vehicle_type found:    {genuine_2_found}")
except NameError:
    print("  (see section 5 above)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. When to Use This — Practical Guidance
# MAGIC
# MAGIC **Use InsuranceEBM when:**
# MAGIC
# MAGIC - You suspect non-linear rating factor relationships (driver age, vehicle age, NCD)
# MAGIC   but have not yet characterised them — EBM will find the shape automatically
# MAGIC - You need the shape functions themselves: the relativities table output is directly
# MAGIC   auditable by a pricing actuary or regulator without post-hoc SHAP or approximation
# MAGIC - Risk ordering (Gini coefficient) matters as much as or more than calibration —
# MAGIC   reinsurance pricing, underwriting scores, portfolio selection
# MAGIC - You want automatic interaction detection to reduce the manual search space for
# MAGIC   GLM interaction terms
# MAGIC - You have >= 5,000 policies in the training set (below this, the boosting procedure
# MAGIC   can overfit individual bins)
# MAGIC
# MAGIC **Use GLM instead when:**
# MAGIC
# MAGIC - Poisson deviance is the primary metric and your GLM is already well-specified (correct
# MAGIC   transformations, main interactions captured): a correctly-specified GLM is close to
# MAGIC   oracle on deviance and EBM adds little
# MAGIC - Exposure calibration accuracy is critical (price-to-burn applications): validate the
# MAGIC   init_score exposure handling on your DGP before production use — there is a known
# MAGIC   calibration scale issue in some EBM configurations
# MAGIC - Fit time is a hard constraint: EBM takes 60-120s on 10k policies, GLM takes <1s
# MAGIC - Regulatory submission requires a factor table with explicit GLM formula: EBM relativities
# MAGIC   can be exported as a lookup table, but this requires a re-certification step
# MAGIC
# MAGIC **Data requirements:**
# MAGIC
# MAGIC - At least 5,000 policies, ideally 20,000+
# MAGIC - Continuous rating factors should not be heavily discretised before fitting — EBM's
# MAGIC   binning does this internally and manual discretisation loses information
# MAGIC - Exposure must be strictly positive (use 1.0 as a proxy if exposure is unknown)
# MAGIC - For interaction detection to be reliable: >= 1,000 policies in each interaction cell
# MAGIC
# MAGIC **On the deviance caveat (from the README):**
# MAGIC
# MAGIC EBM exposure handling via init_score can introduce a calibration scale error on some
# MAGIC DGPs, producing inflated absolute deviance figures without affecting the shape functions
# MAGIC or risk ordering. The Gini coefficient is not affected by this. Use Gini as the primary
# MAGIC comparison metric and validate the calibration separately using a double-lift chart.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *insurance-gam v0.2+ | [GitHub](https://github.com/burning-cost/insurance-gam) | [Burning Cost](https://burning-cost.github.io)*
