# Databricks notebook source
# This script is designed to run on Databricks serverless compute (Free Edition).
# Run via: Databricks UI > Workspace > Import > this file, then attach to serverless cluster.
# Or submit as a job via the Databricks Jobs API.
#
# What this proves: EBM captures non-linear rating factor effects that a standard
# Poisson GLM cannot represent with linear/quadratic terms — U-shaped driver age,
# exponential NCD discount, hard threshold at vehicle age 8. The Gini improvement
# is the production-relevant metric; the deviance comparison is nuanced (see README).
#
# Library: insurance-gam v0.1.6
# Date: 2026-03-22

# COMMAND ----------

# Install dependencies on the cluster
# interpret must be installed before insurance-gam so the optional extra resolves
%pip install "interpret==0.7.0" "scikit-learn>=1.3" "polars>=0.20" "statsmodels" --quiet

# COMMAND ----------

%pip install "insurance-gam[ebm]==0.1.6" --no-deps --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

print("=" * 68)
print("Benchmark: InsuranceEBM vs Poisson GLM — UK Motor Frequency")
print("Library:   insurance-gam v0.1.6")
print("Compute:   Databricks serverless")
print("=" * 68)

# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------
# Synthetic UK motor portfolio with four non-linear effects that a standard
# Poisson GLM with linear terms cannot recover.
#
# Effects:
#   1. Driver age — U-shaped hazard. Young (<25) and old (>65) are riskier.
#      A quadratic term in a GLM can approximate this but only if you know
#      to include it. EBM discovers it data-adaptively.
#   2. NCD years — exponential discount, not linear. Each additional year of
#      no-claims discount is worth less than the last.
#   3. Vehicle age — discrete threshold at age 8 (older vehicles claim more).
#      A linear term underestimates the effect size and places it wrong.
#   4. Annual miles — log-linear. Standard to include in GLMs.
#
# DGP is Poisson with known log-rate, so we can compute oracle deviance.


def make_dataset(n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)

    driver_age   = rng.integers(17, 76, n).astype(float)
    ncd_years    = rng.integers(0, 10, n).astype(float)
    vehicle_age  = rng.integers(0, 16, n).astype(float)
    annual_miles = rng.uniform(3_000, 25_000, n)
    area         = rng.integers(0, 6, n).astype(float)

    # U-shaped driver age: high for young/old, minimum around age 40
    age_effect = (
        0.8 * np.exp(-0.15 * np.maximum(driver_age - 17, 0))
        + 0.3 * np.exp(0.05 * np.maximum(driver_age - 55, 0))
    )
    # Exponential NCD discount (not linear)
    ncd_effect = -0.12 * ncd_years - 0.008 * ncd_years ** 2
    # Hard threshold at vehicle age 8
    veh_effect   = 0.35 * (vehicle_age >= 8).astype(float)
    # Log-miles (standard)
    miles_effect = 0.25 * np.log(annual_miles / 10_000)
    # Area categorical
    area_coef    = np.array([0.0, 0.15, 0.30, -0.10, 0.20, -0.05])
    area_effect  = area_coef[area.astype(int)]

    log_rate = -2.8 + age_effect + ncd_effect + veh_effect + miles_effect + area_effect
    exposure = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(log_rate) * exposure)

    df = pl.DataFrame({
        "driver_age":   driver_age,
        "ncd_years":    ncd_years,
        "vehicle_age":  vehicle_age,
        "annual_miles": annual_miles,
        "area":         area,
    })
    return df, y, exposure, log_rate


df, y, exposure, log_rate = make_dataset(n=10_000, seed=42)

n_train = 7_500
X_train, X_test   = df[:n_train], df[n_train:]
y_train, y_test   = y[:n_train],  y[n_train:]
exp_train, exp_test = exposure[:n_train], exposure[n_train:]

print(f"\nDataset:       {n_train:,} train  /  {len(y_test):,} test observations")
print(f"Mean frequency: {y_train.sum() / exp_train.sum():.4f} claims/year")
print(f"Claim rate range (true): [{np.exp(log_rate).min():.4f}, {np.exp(log_rate).max():.4f}]")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean unit Poisson deviance (lower is better)."""
    eps = 1e-10
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, eps)
    d = 2.0 * (
        np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
        - (y_true - y_pred)
    )
    return float(np.mean(d))


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalised Gini coefficient.

    Measures risk-ordering quality: does the model correctly identify which
    policies will have more claims? A Gini of 0 means the model has no
    discriminatory power. A Gini of 1 means perfect ordering.

    Standard industry metric for pricing model validation.
    """
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    n = len(y_sorted)
    cum_actual = np.cumsum(y_sorted) / (np.sum(y_sorted) + 1e-12)
    cum_pop    = np.arange(1, n + 1) / n
    lorenz_area = float(np.trapz(cum_actual, cum_pop))
    return 2.0 * lorenz_area - 1.0


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, exp: np.ndarray, n_deciles: int = 10) -> pl.DataFrame:
    """
    Actual/expected by predicted decile.

    The gold standard calibration check for a pricing model. Each row is a
    decile of predicted risk, and the A/E ratio tells you whether the model
    is over- or under-predicting that segment.

    A well-calibrated model has A/E close to 1.0 across all deciles.
    A model that is globally calibrated but has monotone A/E across deciles
    is miss-ranking risks — the right volume, wrong distribution.
    """
    order    = np.argsort(y_pred)
    y_s, yp_s, e_s = y_true[order], y_pred[order], exp[order]

    indices  = np.array_split(np.arange(len(y_s)), n_deciles)
    rows = []
    for i, idx in enumerate(indices):
        actual   = y_s[idx].sum()
        expected = yp_s[idx].sum()
        ae       = actual / (expected + 1e-12)
        rows.append({
            "decile":    i + 1,
            "n":         len(idx),
            "actual":    round(float(actual), 2),
            "predicted": round(float(expected), 2),
            "ae_ratio":  round(float(ae), 4),
        })
    return pl.DataFrame(rows)


print("Metric functions defined.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Baseline 1: Poisson GLM with linear + quadratic terms
# ---------------------------------------------------------------------------
# This is what a real pricing team would use today: a Poisson GLM with
# some engineering to handle driver age (quadratic allows a curve, though
# not a full U-shape at both ends), log miles (standard), and area dummies.
# This is a competent, fairly specified baseline — not a strawman.

from sklearn.linear_model import PoissonRegressor


def build_glm_design(df_pl: pl.DataFrame) -> np.ndarray:
    df = df_pl.to_pandas()
    return np.column_stack([
        df["driver_age"],
        df["driver_age"] ** 2 / 1000,          # quadratic (partial non-linearity)
        df["ncd_years"],
        df["vehicle_age"],
        np.log(df["annual_miles"] / 10_000),
        (df["area"] == 1).astype(float),
        (df["area"] == 2).astype(float),
        (df["area"] == 3).astype(float),
        (df["area"] == 4).astype(float),
        (df["area"] == 5).astype(float),
    ])


print("Fitting Poisson GLM (linear + quadratic terms)...")
t0 = time.time()

Xd_train = build_glm_design(X_train)
Xd_test  = build_glm_design(X_test)

# sklearn PoissonRegressor fits rate (claims/exposure) with exposure as weight
y_rate_train = y_train / np.maximum(exp_train, 1e-10)
glm = PoissonRegressor(alpha=0, max_iter=1000)
glm.fit(Xd_train, y_rate_train, sample_weight=exp_train)

glm_pred  = glm.predict(Xd_test) * exp_test
glm_dev   = poisson_deviance(y_test, glm_pred)
glm_gini  = gini_coefficient(y_test, glm_pred)
glm_cal   = calibration_table(y_test, glm_pred, exp_test)
glm_time  = time.time() - t0

print(f"  Fit time:         {glm_time:.1f}s")
print(f"  Poisson deviance: {glm_dev:.6f}")
print(f"  Gini:             {glm_gini:.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Baseline 2: Oracle (true DGP)
# ---------------------------------------------------------------------------
# The oracle uses the true log-rate from the DGP — the theoretical best
# any model could achieve given data of this size. Everything above oracle
# deviance is sampling noise.

true_pred_test = np.exp(log_rate[n_train:]) * exp_test
oracle_dev  = poisson_deviance(y_test, true_pred_test)
oracle_gini = gini_coefficient(y_test, true_pred_test)

print(f"Oracle (true DGP) Poisson deviance: {oracle_dev:.6f}")
print(f"Oracle Gini:                         {oracle_gini:.4f}")
print(f"GLM gap from oracle (deviance):      {glm_dev - oracle_dev:.6f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# InsuranceEBM
# ---------------------------------------------------------------------------
# EBM learns each feature's shape function independently via boosting, then
# adds pairwise interactions. No feature engineering required — the U-shape,
# the threshold at vehicle_age=8, and the exponential NCD discount are all
# recovered from data. The output is a set of shape functions that a pricing
# team can inspect, challenge, and apply monotonicity constraints to.

from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

print("\nFitting InsuranceEBM (loss=poisson, interactions='3x')...")
print("(This takes 60-120s on serverless compute — EBM bagging is single-threaded)")
t0 = time.time()

ebm = InsuranceEBM(
    loss="poisson",
    interactions="3x",  # up to 3 pairwise interaction terms
    random_state=42,
)
ebm.fit(X_train, y_train, exposure=exp_train)

ebm_pred  = ebm.predict(X_test, exposure=exp_test)
ebm_dev   = poisson_deviance(y_test, ebm_pred)
ebm_gini  = gini_coefficient(y_test, ebm_pred)
ebm_cal   = calibration_table(y_test, ebm_pred, exp_test)
ebm_time  = time.time() - t0

print(f"  Fit time:         {ebm_time:.1f}s")
print(f"  Poisson deviance: {ebm_dev:.6f}")
print(f"  Gini:             {ebm_gini:.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Interpretability: relativities table
# ---------------------------------------------------------------------------
# This is where EBM earns its keep over a GBM: you can read off exactly what
# the model thinks about each rating factor, in the same format as a traditional
# GLM relativities table. Every pricing actuary knows how to work with this.

rt = RelativitiesTable(ebm)
print("\nNCD years relativities (EBM learned shape):")
print(rt.table("ncd_years").to_pandas().to_string(index=False))
print("\nSummary of all features:")
print(rt.summary().to_pandas().to_string(index=False))

# COMMAND ----------

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

dev_improvement  = 100.0 * (glm_dev - ebm_dev)  / max(abs(glm_dev),  1e-12)
gini_improvement = 100.0 * (ebm_gini - glm_gini) / max(abs(glm_gini), 1e-12)
glm_oracle_gap   = glm_dev - oracle_dev
ebm_oracle_gap   = ebm_dev - oracle_dev

print("\n" + "=" * 68)
print("RESULTS — test set")
print("=" * 68)
print(f"\n{'Model':<30} {'Deviance':>12} {'Gini':>10} {'Dev gap (oracle)':>18}")
print("-" * 72)
print(f"{'Oracle (true DGP)':<30} {oracle_dev:>12.6f} {oracle_gini:>10.4f} {'—':>18}")
print(f"{'Poisson GLM (lin+quad)':<30} {glm_dev:>12.6f} {glm_gini:>10.4f} {glm_oracle_gap:>18.6f}")
print(f"{'InsuranceEBM':<30} {ebm_dev:>12.6f} {ebm_gini:>10.4f} {ebm_oracle_gap:>18.6f}")
print("-" * 72)
print(f"\nEBM vs GLM — Gini improvement:     {gini_improvement:+.1f}%")
print(f"EBM vs GLM — Deviance improvement: {dev_improvement:+.1f}%")

print("\n--- Calibration table: Poisson GLM ---")
print(glm_cal.to_pandas().to_string(index=False))

print("\n--- Calibration table: InsuranceEBM ---")
print(ebm_cal.to_pandas().to_string(index=False))

# COMMAND ----------

# ---------------------------------------------------------------------------
# Honest interpretation
# ---------------------------------------------------------------------------

print("\n" + "=" * 68)
print("INTERPRETATION")
print("=" * 68)

if ebm_oracle_gap < glm_oracle_gap:
    print(f"\nEBM is CLOSER to oracle on deviance ({ebm_oracle_gap:.4f} vs {glm_oracle_gap:.4f}).")
    print("EBM has successfully learned the non-linear DGP structure.")
else:
    print(f"\nGLM is closer to oracle on deviance ({glm_oracle_gap:.4f}) than EBM ({ebm_oracle_gap:.4f}).")
    print("This occurs when the GLM's quadratic term sufficiently approximates")
    print("the DGP non-linearities, or when EBM calibration via init_score offset")
    print("introduces a scale error. The Gini comparison is unaffected by this.")
    print("See README Performance section for a full discussion of this trade-off.")

print(f"\nGini comparison: EBM={ebm_gini:.4f} vs GLM={glm_gini:.4f}")
if ebm_gini > glm_gini:
    print("EBM has better risk ORDERING — it ranks which policies are riskier")
    print("more accurately than the GLM. For relativities and segmentation this")
    print("is the more important metric.")
else:
    print("GLM has better risk ordering on this run — likely because the DGP's")
    print("non-linearities are approximately polynomial and the quadratic term")
    print("captures most of the signal.")

print("\nFit times:")
print(f"  GLM:  {glm_time:.1f}s")
print(f"  EBM:  {ebm_time:.1f}s")
print("\nEBM is significantly slower. The payoff is shape functions you can")
print("inspect, challenge, and apply constraints to — without SHAP post-hoc.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Exit with results dict (for job orchestration)
# ---------------------------------------------------------------------------

results = {
    "oracle_deviance":   float(oracle_dev),
    "oracle_gini":       float(oracle_gini),
    "glm_deviance":      float(glm_dev),
    "glm_gini":          float(glm_gini),
    "glm_oracle_gap":    float(glm_oracle_gap),
    "ebm_deviance":      float(ebm_dev),
    "ebm_gini":          float(ebm_gini),
    "ebm_oracle_gap":    float(ebm_oracle_gap),
    "gini_improvement_pct":    float(gini_improvement),
    "deviance_improvement_pct": float(dev_improvement),
    "glm_fit_seconds":   float(glm_time),
    "ebm_fit_seconds":   float(ebm_time),
    "n_train":           n_train,
    "n_test":            len(y_test),
}

print("\nRaw results JSON (for job output capture):")
print(json.dumps(results, indent=2))

dbutils.notebook.exit(json.dumps(results))
