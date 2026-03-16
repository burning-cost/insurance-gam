# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: InsuranceEBM vs Poisson GLM for UK Motor Frequency
# MAGIC
# MAGIC **Library:** insurance-gam v0.1.3
# MAGIC
# MAGIC **Date:** 2026-03-16

# COMMAND ----------

# Install interpret first to ensure EBM is available, then install the library
%pip install "interpret==0.7.0" "scikit-learn" "polars" --quiet

# COMMAND ----------

%pip install "git+https://github.com/burning-cost/insurance-gam.git" --no-deps --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import polars as pl
import warnings
warnings.filterwarnings("ignore")

# Verify interpret is working
from interpret.glassbox import ExplainableBoostingRegressor
print("interpret loaded, EBM class:", ExplainableBoostingRegressor)

def make_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    driver_age    = rng.integers(17, 76, n).astype(float)
    ncd_years     = rng.integers(0, 10, n).astype(float)
    vehicle_age   = rng.integers(0, 16, n).astype(float)
    annual_miles  = rng.uniform(3_000, 25_000, n)
    area          = rng.integers(0, 6, n).astype(float)
    age_effect = (0.8 * np.exp(-0.15 * np.maximum(driver_age - 17, 0))
                  + 0.3 * np.exp(0.05 * np.maximum(driver_age - 55, 0)))
    ncd_effect = -0.12 * ncd_years - 0.008 * ncd_years ** 2
    veh_effect = 0.35 * (vehicle_age >= 8).astype(float)
    miles_effect = 0.25 * np.log(annual_miles / 10_000)
    area_coef = np.array([0.0, 0.15, 0.30, -0.10, 0.20, -0.05])
    area_effect = area_coef[area.astype(int)]
    log_rate = -2.8 + age_effect + ncd_effect + veh_effect + miles_effect + area_effect
    exposure = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(log_rate) * exposure)
    return driver_age, ncd_years, vehicle_age, annual_miles, area, y, exposure, log_rate

def poisson_deviance(y_true, y_pred):
    eps = 1e-10
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, eps)
    d = 2.0 * (np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0) - (y_true - y_pred))
    return float(np.mean(d))

def gini_coefficient(y_true, y_pred):
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    n = len(y_sorted)
    cum_actual = np.cumsum(y_sorted) / np.sum(y_sorted)
    cum_pop = np.arange(1, n + 1) / n
    return float(2.0 * np.trapz(cum_actual, cum_pop) - 1.0)

driver_age, ncd_years, vehicle_age, annual_miles, area, y, exposure, log_rate = make_dataset()
n_train = 4500

X_pl = pl.DataFrame({"driver_age": driver_age, "ncd_years": ncd_years,
                      "vehicle_age": vehicle_age, "annual_miles": annual_miles, "area": area})
X_train_pl, X_test_pl = X_pl[:n_train], X_pl[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
exp_train, exp_test = exposure[:n_train], exposure[n_train:]
print(f"Dataset: {n_train} train / {6000-n_train} test. Mean freq: {y_train.sum()/exp_train.sum():.4f}")

# COMMAND ----------

from sklearn.linear_model import PoissonRegressor

def build_design(df_pl):
    df = df_pl.to_pandas()
    return np.column_stack([
        df["driver_age"], df["driver_age"]**2/1000, df["ncd_years"], df["vehicle_age"],
        np.log(df["annual_miles"]/10_000),
        (df["area"]==1).astype(float), (df["area"]==2).astype(float),
        (df["area"]==3).astype(float), (df["area"]==4).astype(float), (df["area"]==5).astype(float),
    ])

Xd_train = build_design(X_train_pl)
Xd_test  = build_design(X_test_pl)
y_rate_train = y_train / np.maximum(exp_train, 1e-10)
glm = PoissonRegressor(alpha=0, max_iter=1000)
glm.fit(Xd_train, y_rate_train, sample_weight=exp_train)
glm_pred = glm.predict(Xd_test) * exp_test
glm_dev  = poisson_deviance(y_test, glm_pred)
glm_gini = gini_coefficient(y_test, glm_pred)
print(f"GLM Poisson deviance: {glm_dev:.6f}, Gini: {glm_gini:.4f}")

# COMMAND ----------

from insurance_gam.ebm import InsuranceEBM
print("Fitting InsuranceEBM...")
ebm = InsuranceEBM(loss="poisson", interactions="3x", random_state=42)
ebm.fit(X_train_pl, y_train, exposure=exp_train)
ebm_pred = ebm.predict(X_test_pl, exposure=exp_test)
ebm_dev  = poisson_deviance(y_test, ebm_pred)
ebm_gini = gini_coefficient(y_test, ebm_pred)
print(f"EBM Poisson deviance: {ebm_dev:.6f}, Gini: {ebm_gini:.4f}")

# COMMAND ----------

true_pred_test = np.exp(log_rate[n_train:]) * exp_test
oracle_dev  = poisson_deviance(y_test, true_pred_test)
oracle_gini = gini_coefficient(y_test, true_pred_test)
dev_improvement  = 100 * (glm_dev - ebm_dev) / glm_dev
gini_improvement = 100 * (ebm_gini - glm_gini) / max(abs(glm_gini), 1e-6)

print(f"\nOracle deviance:  {oracle_dev:.6f}")
print(f"GLM deviance:     {glm_dev:.6f}  (gap: {glm_dev - oracle_dev:.6f})")
print(f"EBM deviance:     {ebm_dev:.6f}  (gap: {ebm_dev - oracle_dev:.6f})")
print(f"EBM improvement:  {dev_improvement:+.1f}% deviance, {gini_improvement:+.1f}% Gini")

# COMMAND ----------

import json as _json
_results = {
    "glm_poisson_deviance": float(glm_dev),
    "glm_gini": float(glm_gini),
    "ebm_poisson_deviance": float(ebm_dev),
    "ebm_gini": float(ebm_gini),
    "oracle_poisson_deviance": float(oracle_dev),
    "oracle_gini": float(oracle_gini),
    "deviance_improvement_pct": float(dev_improvement),
    "gini_improvement_pct": float(gini_improvement),
    "ebm_gap_from_oracle": float(ebm_dev - oracle_dev),
    "glm_gap_from_oracle": float(glm_dev - oracle_dev),
}
dbutils.notebook.exit(_json.dumps(_results))
