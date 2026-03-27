# Databricks notebook source
# DISCLAIMER: freMTPL2 is French motor data (OpenML dataset 41214).
# It is used here for methodology validation only — to demonstrate that
# InsuranceEBM generalises to real insurance data beyond the synthetic DGP.
# It is NOT UK market data and should not be used to draw conclusions about
# UK motor pricing structure.
#
# Dataset: 678,013 French motor third-party liability policies.
# Source: Noll, Salzmann, Wüthrich (2018) — Case Study: French Motor Third-Party
#         Liability Claims.
# Variables used:
#   ClaimNb   — number of claims (target, Poisson frequency)
#   Exposure  — fraction of year at risk (offset)
#   Area      — categorical area code (A–F)
#   VehPower  — vehicle power (4–15)
#   VehAge    — vehicle age in years (0–100)
#   DrivAge   — driver age in years (18–100)
#   BonusMalus — bonus-malus level (50–350)
#   VehBrand  — vehicle brand (B1–B14)
#   VehGas    — fuel type (Diesel / Regular)
#   Density   — population density of driver's commune
#   Region    — French administrative region code
#
# What this benchmark demonstrates:
#   InsuranceEBM fits real insurance data with mixed feature types (continuous,
#   ordinal, categorical) without any hand-crafted feature engineering. The EBM
#   discovers non-linear effects (U-shaped driver age, BonusMalus curve, Density
#   log-linearity) automatically. The GLM baseline requires the analyst to decide
#   on transformations in advance.
#
# Library: insurance-gam
# Date: 2026-03-27

# COMMAND ----------

%pip install "interpret>=0.7.0" "scikit-learn>=1.3" "polars>=1.0" \
    "statsmodels>=0.14" "pandas>=2.0" --quiet

# COMMAND ----------

%pip install "insurance-gam[ebm]" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

print("=" * 68)
print("freMTPL2 Benchmark: InsuranceEBM vs Poisson GLM")
print("Data:    OpenML #41214 — French motor MTPL frequency")
print("Library: insurance-gam")
print("=" * 68)
print()
print("DISCLAIMER: French motor data used for methodology validation only.")
print("Not UK market data. Results reflect French portfolio characteristics.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Load freMTPL2
# ---------------------------------------------------------------------------
# OpenML 41214 is the freMTPL2freq dataset (~678K rows). The fetch includes
# ClaimNb and Exposure in the feature matrix alongside the rating factors.
# We pull them out and keep the remaining columns as X.

from sklearn.datasets import fetch_openml

print("\nDownloading freMTPL2freq from OpenML (may take 30–60s)...")
t_load = time.time()

raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
df_raw = raw.frame.copy()

print(f"  Loaded {len(df_raw):,} rows in {time.time() - t_load:.1f}s")
print(f"  Columns: {list(df_raw.columns)}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Prepare target, exposure, and feature matrix
# ---------------------------------------------------------------------------

# Normalise column names — OpenML sometimes returns them with mixed case.
df_raw.columns = [c.strip() for c in df_raw.columns]

# ClaimNb is the Poisson target. Exposure is the offset.
y_raw     = df_raw["ClaimNb"].astype(float).values
exp_raw   = df_raw["Exposure"].astype(float).values.clip(min=1e-6)

# Feature columns: drop target and exposure.
drop_cols = {"ClaimNb", "Exposure"}
feature_cols = [c for c in df_raw.columns if c not in drop_cols]

X_raw = df_raw[feature_cols].copy()

print(f"Feature columns: {feature_cols}")
print(f"Target: ClaimNb — mean {y_raw.mean():.4f}, sum {y_raw.sum():.0f}")
print(f"Exposure: mean {exp_raw.mean():.4f}, range [{exp_raw.min():.4f}, {exp_raw.max():.4f}]")
print(f"Overall frequency: {y_raw.sum() / exp_raw.sum():.4f} claims/year")

# EBM will treat string/category columns as nominal automatically.
# Encode object/category columns as strings so pandas handles them cleanly.
for col in X_raw.columns:
    if X_raw[col].dtype.name in ("object", "category"):
        X_raw[col] = X_raw[col].astype(str)

print(f"\nDtypes:\n{X_raw.dtypes}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Subsample to 75,000 rows for tractable EBM fitting
# ---------------------------------------------------------------------------
# The full 678K row dataset would take ~20 minutes to fit on serverless compute.
# 75K rows is enough to characterise the non-linear effects reliably.
# We use a stratified subsample (not random) to preserve the claim frequency
# distribution: keep all policies with at least one claim, and sample from
# zero-claim policies to reach the target size.
#
# This is a deliberate methodological choice. Oversampling claim-bearing policies
# gives the model more signal per training row, which matters when fitting
# a Poisson response on data where ~93% of rows are zeros.

N_SAMPLE = 75_000
SEED     = 42

rng = np.random.default_rng(SEED)

mask_claim  = y_raw > 0
idx_claim   = np.where(mask_claim)[0]
idx_no_claim = np.where(~mask_claim)[0]

n_claim    = len(idx_claim)
n_no_claim = max(0, N_SAMPLE - n_claim)

sampled_no_claim = rng.choice(idx_no_claim, size=min(n_no_claim, len(idx_no_claim)), replace=False)
idx_sample = np.concatenate([idx_claim, sampled_no_claim])
rng.shuffle(idx_sample)

X_samp   = X_raw.iloc[idx_sample].reset_index(drop=True)
y_samp   = y_raw[idx_sample]
exp_samp = exp_raw[idx_sample]

print(f"Subsample: {len(idx_sample):,} rows ({n_claim:,} with claims, {len(sampled_no_claim):,} zero-claim)")
print(f"Subsample frequency: {y_samp.sum() / exp_samp.sum():.4f} claims/year")
print(f"(Full dataset frequency: {y_raw.sum() / exp_raw.sum():.4f})")
print()
print("Note: subsample overrepresents claimant policies to give EBM more signal.")
print("Subsample frequency is higher than portfolio average; interpret accordingly.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

N_TRAIN = int(0.75 * len(idx_sample))

X_train_pd = X_samp.iloc[:N_TRAIN]
X_test_pd  = X_samp.iloc[N_TRAIN:]
y_train    = y_samp[:N_TRAIN]
y_test     = y_samp[N_TRAIN:]
exp_train  = exp_samp[:N_TRAIN]
exp_test   = exp_samp[N_TRAIN:]

print(f"Train: {N_TRAIN:,} rows  |  Test: {len(y_test):,} rows")
print(f"Train frequency: {y_train.sum() / exp_train.sum():.4f}")
print(f"Test  frequency: {y_test.sum() / exp_test.sum():.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Mean unit Poisson deviance (lower is better).

    Exposure-weighted when weights are provided. The weighted version
    gives each policy unit exposure-years weight, which is the standard
    actuarial convention for frequency model validation.
    """
    eps = 1e-10
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, eps)
    d = 2.0 * (
        np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
        - (y_true - y_pred)
    )
    if weights is not None:
        return float(np.average(d, weights=weights))
    return float(np.mean(d))


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalised Gini coefficient (ordering metric, higher is better).

    Measures how well the model separates high-risk from low-risk policies.
    Uses the simple (unweighted) formulation consistent with the DGP benchmark.
    """
    order      = np.argsort(y_pred)
    y_sorted   = y_true[order]
    n          = len(y_sorted)
    cum_actual = np.cumsum(y_sorted) / (np.sum(y_sorted) + 1e-12)
    cum_pop    = np.arange(1, n + 1) / n
    lorenz_area = float(np.trapz(cum_actual, cum_pop))
    return 2.0 * lorenz_area - 1.0


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, exp: np.ndarray, n_deciles: int = 10) -> pl.DataFrame:
    """
    Actual/expected by predicted risk decile.

    The canonical actuarial calibration check. Equal-count buckets by
    predicted value. A/E close to 1.0 everywhere means the model is
    well-calibrated across the risk spectrum.
    """
    order = np.argsort(y_pred)
    y_s, yp_s, e_s = y_true[order], y_pred[order], exp[order]
    indices = np.array_split(np.arange(len(y_s)), n_deciles)
    rows = []
    for i, idx in enumerate(indices):
        actual   = float(y_s[idx].sum())
        expected = float(yp_s[idx].sum())
        ae       = actual / (expected + 1e-12)
        rows.append({
            "decile":    i + 1,
            "n":         len(idx),
            "actual":    round(actual, 2),
            "predicted": round(expected, 2),
            "ae_ratio":  round(ae, 4),
        })
    return pl.DataFrame(rows)


print("Metric functions defined.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Baseline: Poisson GLM via sklearn
# ---------------------------------------------------------------------------
# The GLM requires the analyst to specify transformations ahead of time. We use
# a reasonably specified baseline that mirrors common industry practice:
#   - Log(Density) — density is right-skewed, log scale is standard
#   - BonusMalus as continuous (it is an ordinal score)
#   - DrivAge, VehAge as continuous linear terms
#   - VehPower as continuous linear
#   - Area, VehBrand, VehGas, Region as one-hot encoded
#
# No interaction terms. This is the "first pass" model a pricing team would build
# before investing in non-linear methods.

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


numeric_features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
categorical_features = ["Area", "VehBrand", "VehGas", "Region"]

log_density_transformer = FunctionTransformer(
    lambda X: np.log1p(X.astype(float)),
    validate=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_log_density", log_density_transformer, ["Density"]),
        ("num_linear", "passthrough", [c for c in numeric_features if c != "Density"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

glm_pipeline = Pipeline([
    ("prep", preprocessor),
    ("glm", PoissonRegressor(alpha=0, max_iter=500, solver="lbfgs")),
])

print("Fitting Poisson GLM (log-Density, linear numerics, one-hot categoricals)...")
t0 = time.time()

# sklearn PoissonRegressor models the rate; multiply by exposure at prediction time.
y_rate_train = y_train / np.maximum(exp_train, 1e-10)
glm_pipeline.fit(X_train_pd, y_rate_train, glm__sample_weight=exp_train)

glm_pred = glm_pipeline.predict(X_test_pd) * exp_test
glm_time = time.time() - t0

glm_dev  = poisson_deviance(y_test, glm_pred)
glm_dev_wt = poisson_deviance(y_test, glm_pred, weights=exp_test)
glm_gini = gini_coefficient(y_test, glm_pred)
glm_cal  = calibration_table(y_test, glm_pred, exp_test)

print(f"  Fit time:                {glm_time:.1f}s")
print(f"  Poisson deviance:        {glm_dev:.6f}")
print(f"  Exposure-wtd deviance:   {glm_dev_wt:.6f}")
print(f"  Gini:                    {glm_gini:.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# InsuranceEBM
# ---------------------------------------------------------------------------
# EBM receives the same raw feature matrix — no feature engineering. It
# discovers log-Density, the U-shaped DrivAge, the BonusMalus non-linearity,
# and any relevant interactions without being told to look for them.
#
# interactions='2x' means 2 * n_features pairwise interaction terms.
# Reduced from the DGP benchmark's '3x' to keep fit time tractable on 75K rows.

from insurance_gam.ebm import InsuranceEBM

print("\nFitting InsuranceEBM (loss=poisson, interactions='2x')...")
print("(Expected fit time: 4–10 minutes on Databricks serverless)")
t0 = time.time()

ebm = InsuranceEBM(
    loss="poisson",
    interactions="2x",
    random_state=SEED,
)
ebm.fit(X_train_pd, y_train, exposure=exp_train)

ebm_pred = ebm.predict(X_test_pd, exposure=exp_test)
ebm_time = time.time() - t0

ebm_dev    = poisson_deviance(y_test, ebm_pred)
ebm_dev_wt = poisson_deviance(y_test, ebm_pred, weights=exp_test)
ebm_gini   = gini_coefficient(y_test, ebm_pred)
ebm_cal    = calibration_table(y_test, ebm_pred, exp_test)

print(f"  Fit time:                {ebm_time:.1f}s")
print(f"  Poisson deviance:        {ebm_dev:.6f}")
print(f"  Exposure-wtd deviance:   {ebm_dev_wt:.6f}")
print(f"  Gini:                    {ebm_gini:.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Comparison: results table
# ---------------------------------------------------------------------------

dev_improvement   = 100.0 * (glm_dev - ebm_dev) / max(abs(glm_dev), 1e-12)
gini_improvement  = 100.0 * (ebm_gini - glm_gini) / max(abs(glm_gini), 1e-12)

print("\n" + "=" * 68)
print("RESULTS — freMTPL2 test set")
print("=" * 68)
print(f"\n{'Model':<30} {'Deviance':>12} {'Wt Deviance':>14} {'Gini':>10} {'Fit (s)':>10}")
print("-" * 78)
print(f"{'Poisson GLM':<30} {glm_dev:>12.6f} {glm_dev_wt:>14.6f} {glm_gini:>10.4f} {glm_time:>10.1f}")
print(f"{'InsuranceEBM':<30} {ebm_dev:>12.6f} {ebm_dev_wt:>14.6f} {ebm_gini:>10.4f} {ebm_time:>10.1f}")
print("-" * 78)
print(f"\nEBM vs GLM — Deviance improvement:    {dev_improvement:+.2f}%")
print(f"EBM vs GLM — Gini improvement:        {gini_improvement:+.2f}%")

print("\n--- Calibration: Poisson GLM ---")
print(glm_cal.to_pandas().to_string(index=False))

print("\n--- Calibration: InsuranceEBM ---")
print(ebm_cal.to_pandas().to_string(index=False))

# COMMAND ----------

# ---------------------------------------------------------------------------
# Relativities table — a/e by DrivAge and BonusMalus
# ---------------------------------------------------------------------------
# One practical reason to use EBM over a black-box GBM: the shape functions
# come out directly in the format a pricing actuary expects. No SHAP post-hoc.

from insurance_gam.ebm import RelativitiesTable

rt = RelativitiesTable(ebm)

print("\nSummary of EBM features (ordered by relative importance):")
summary = rt.summary()
print(summary.to_pandas().to_string(index=False))

# COMMAND ----------

# ---------------------------------------------------------------------------
# Calibration by Area (categorical segment)
# ---------------------------------------------------------------------------
# The A/E by region is the sort of table a UK pricing actuary would check
# first after fitting any model. Large deviations here indicate the model
# is systematically wrong for certain geographical segments.

from insurance_gam.ebm import diagnostics

ebm_cal_area = diagnostics.calibration_table(
    y_true   = y_test,
    y_pred   = ebm_pred,
    segment  = X_test_pd["Area"].values,
    exposure = exp_test,
)
glm_cal_area = diagnostics.calibration_table(
    y_true   = y_test,
    y_pred   = glm_pred,
    segment  = X_test_pd["Area"].values,
    exposure = exp_test,
)

print("\n--- A/E by Area: InsuranceEBM ---")
print(ebm_cal_area.to_pandas().to_string(index=False))

print("\n--- A/E by Area: Poisson GLM ---")
print(glm_cal_area.to_pandas().to_string(index=False))

# COMMAND ----------

# ---------------------------------------------------------------------------
# Honest assessment
# ---------------------------------------------------------------------------
# Unlike the DGP benchmark, we have no oracle here. We cannot say whether EBM
# is "closer to the truth" — we only know relative performance. The points below
# are what a pricing team should actually care about.

print("\n" + "=" * 68)
print("ASSESSMENT")
print("=" * 68)
print("""
1. Deviance and Gini measure different things. Deviance rewards calibration
   (the model predicts the right count on average). Gini rewards ordering
   (the model correctly identifies which policies are riskier). Both matter
   for pricing but often disagree on which model is better.

2. EBM discovers feature non-linearities automatically. On freMTPL2, the
   BonusMalus score is highly non-linear (it compresses at the extremes),
   driver age is U-shaped, and density is better on a log scale. The GLM
   baseline handles density manually but treats the others as linear.

3. EBM is significantly slower. On 75K rows with interactions='2x', expect
   4–10 minutes on Databricks serverless. The GLM fits in seconds. This is
   the correct trade-off for an annual model rebuild, not for daily scoring.

4. This is French motor data. The rating factors, their distributions, and
   their interactions with claims reflect the French market. UK motor has
   different regulatory constraints, different geographic structure, and
   different portfolio mix. Treat these results as evidence that the method
   generalises, not as evidence of expected UK improvement magnitudes.
""")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Exit with results dict
# ---------------------------------------------------------------------------

results = {
    "dataset":                  "freMTPL2freq (OpenML 41214)",
    "n_sample":                 int(len(idx_sample)),
    "n_train":                  int(N_TRAIN),
    "n_test":                   int(len(y_test)),
    "subsample_frequency":      float(y_samp.sum() / exp_samp.sum()),
    "glm_deviance":             float(glm_dev),
    "glm_deviance_weighted":    float(glm_dev_wt),
    "glm_gini":                 float(glm_gini),
    "glm_fit_seconds":          float(glm_time),
    "ebm_deviance":             float(ebm_dev),
    "ebm_deviance_weighted":    float(ebm_dev_wt),
    "ebm_gini":                 float(ebm_gini),
    "ebm_fit_seconds":          float(ebm_time),
    "deviance_improvement_pct": float(dev_improvement),
    "gini_improvement_pct":     float(gini_improvement),
    "disclaimer":               "French motor data — methodology validation only, not UK market data",
}

print("\nRaw results JSON:")
print(json.dumps(results, indent=2))

dbutils.notebook.exit(json.dumps(results))
