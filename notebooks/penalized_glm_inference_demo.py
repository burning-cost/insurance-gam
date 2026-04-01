# Databricks notebook source
# MAGIC %md
# MAGIC # PenalizedGLMInference — Demo
# MAGIC
# MAGIC **Paper**: Manna, Huang, Dey, Gu & He (2025). "Interval Estimation of Coefficients
# MAGIC in Penalized Regression Models of Insurance Data." *Applied Stochastic Models in
# MAGIC Business and Industry*. arXiv:2410.01008.
# MAGIC
# MAGIC **Problem**: After Lasso selects rating factors, naive Wald confidence intervals
# MAGIC on those coefficients have 70-80% actual coverage at the nominal 95% level.
# MAGIC The Lasso biases estimates toward zero — small when the true coefficient is small,
# MAGIC but substantial for large coefficients (age bands, NCD categories, etc.).
# MAGIC
# MAGIC **This notebook demonstrates**:
# MAGIC 1. Simulating a realistic motor frequency dataset (Poisson, log link, exposure)
# MAGIC 2. Fitting `PenalizedGLMInference` and comparing both CI strategies
# MAGIC 3. Forest plots of debiased vs penalized estimates
# MAGIC 4. Gamma severity model (demonstrates multi-family support)
# MAGIC 5. Coverage simulation (empirical validation of the correction)

# COMMAND ----------

# MAGIC %pip install insurance-gam>=0.5.0 statsmodels>=0.14.5

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_gam.penalized_glm_inference import PenalizedGLMInference

print("insurance-gam PenalizedGLMInference demo — starting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic motor frequency dataset
# MAGIC
# MAGIC We simulate a Poisson frequency model with 8 rating factors (age band, NCD,
# MAGIC vehicle group, region, etc.) and 4 noise features. True log-coefficients are
# MAGIC realistic for UK motor: age effect ~0.4, NCD ~0.3, region ~0.2.

# COMMAND ----------

rng = np.random.default_rng(42)
n = 10_000
p = 12  # 4 true, 8 noise

# Simulate rating factor matrix
X = rng.standard_normal((n, p))
feature_names = (
    ["age_band", "ncd_group", "vehicle_group", "region"]
    + [f"noise_{i}" for i in range(8)]
)

# True coefficients: first 4 are non-zero
true_coefs = np.array([0.40, 0.30, 0.20, -0.15] + [0.0] * 8)

# Exposure: policy years, ranging 0.2 to 1.2
exposure = rng.uniform(0.2, 1.2, size=n)

# Poisson frequency
eta = X @ true_coefs + np.log(exposure)
y = rng.poisson(np.exp(eta))

print(f"Dataset: n={n:,}, p={p}, expected claims={y.mean():.4f}/policy-year")
print(f"Observed zeros: {(y == 0).sum():,} / {n:,} ({100*(y==0).mean():.1f}%)")

X_df = pd.DataFrame(X, columns=feature_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit PenalizedGLMInference
# MAGIC
# MAGIC We use `alpha=0.0` so the regularisation strength is chosen by 5-fold CV.
# MAGIC `l1_ratio=1.0` gives pure Lasso (maximum sparsity).

# COMMAND ----------

m = PenalizedGLMInference(
    family="poisson",
    alpha=0.0,           # CV to select lambda
    l1_ratio=1.0,        # pure Lasso
    cv_folds=5,
    random_state=42,
).fit(X_df, y, exposure=exposure)

print(f"Lambda chosen by CV: {m.lambda_:.6f}")
print(f"Features selected:   {m.selected_features_.sum()} / {p}")
print(f"Selected names:      {[n for n, s in zip(feature_names, m.selected_features_) if s]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Asymptotic debiased CIs (Strategy A)
# MAGIC
# MAGIC Fast one-step Newton correction. For n=10,000 and p=12, p^2/n = 144/10000 = 0.014 —
# MAGIC well within the validity condition.

# COMMAND ----------

df_asym = m.confidence_intervals(alpha=0.05)

print("Asymptotic (debiased) CIs:")
print(
    df_asym[df_asym["selected"]][
        ["feature", "coef_penalized", "coef", "se", "ci_lower", "ci_upper", "pvalue"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bootstrap CIs (Strategy B)
# MAGIC
# MAGIC Pearson residual bootstrap with 500 resamples. Takes ~30s for n=10,000.
# MAGIC Bootstrap CIs use the pivot form to correct for the downward bias of the
# MAGIC penalized estimates.

# COMMAND ----------

df_boot = m.bootstrap_ci(alpha=0.05, n_bootstrap=500, random_state=42)

print("Bootstrap CIs (pivot form):")
print(
    df_boot[df_boot["selected"]][
        ["feature", "coef_penalized", "coef", "ci_lower", "ci_upper"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparison: penalized vs debiased vs true coefficient

# COMMAND ----------

comparison_rows = []
for name, true_val, is_sel in zip(feature_names, true_coefs, m.selected_features_):
    if not is_sel:
        continue
    pen = float(df_asym[df_asym["feature"] == name]["coef_penalized"].values[0])
    deb = float(df_asym[df_asym["feature"] == name]["coef"].values[0])
    comparison_rows.append({
        "feature": name,
        "true_coef": true_val,
        "penalized": pen,
        "debiased": deb,
        "bias_penalized": pen - true_val,
        "bias_debiased": deb - true_val,
    })

comp_df = pd.DataFrame(comparison_rows)
print("Bias comparison (selected features only):")
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Forest plots

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Asymptotic
m.forest_plot(strategy="asymptotic", alpha=0.05, ax=axes[0])
axes[0].set_title("Debiased CIs (Strategy A)")

# Bootstrap
m.forest_plot(strategy="bootstrap", n_bootstrap=200, alpha=0.05, ax=axes[1])
axes[1].set_title("Bootstrap CIs (Strategy B)")

# Add true coefficient markers
true_dict = dict(zip(feature_names, true_coefs))
selected_names = [n for n, s in zip(feature_names, m.selected_features_) if s]
for ax in axes:
    for i, name in enumerate(selected_names):
        tv = true_dict.get(name, None)
        if tv is not None:
            ax.scatter([tv], [i], marker="|", color="red", s=100, zorder=5, linewidth=2)

axes[0].scatter([], [], marker="|", color="red", s=100, label="True coef")
axes[0].legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/penalized_glm_forest_plots.png", dpi=150, bbox_inches="tight")
print("Forest plots saved to /tmp/penalized_glm_forest_plots.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Gamma severity model
# MAGIC
# MAGIC Same feature matrix, now with a log-normal-like Gamma response
# MAGIC (claim severity). `PenalizedGLMInference` handles dispersion estimation
# MAGIC automatically.

# COMMAND ----------

shape = 3.0
mu_sev = np.exp(X @ np.array([0.4, 0.3, 0.2, -0.15] + [0.0] * 8) + 7.5)  # £ scale
y_sev = rng.gamma(shape, mu_sev / shape)

m_sev = PenalizedGLMInference(
    family="gamma",
    alpha=0.0,
    l1_ratio=1.0,
    cv_folds=5,
    random_state=42,
).fit(X_df, y_sev)

print(f"Gamma model dispersion (phi): {m_sev.phi_:.4f}  (true 1/shape = {1/shape:.4f})")
df_sev = m_sev.confidence_intervals(alpha=0.05)
print(
    df_sev[df_sev["selected"]][
        ["feature", "coef_penalized", "coef", "ci_lower", "ci_upper"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Coverage simulation (Monte Carlo)
# MAGIC
# MAGIC Run 200 simulation trials.  In each trial, generate a fresh dataset and check
# MAGIC whether the debiased CI for the `age_band` coefficient (true value = 0.4) contains
# MAGIC the true value.  Naive Wald CI around the penalized estimate is also computed for
# MAGIC comparison.  Target: debiased coverage ~95%, naive coverage ~70-80%.

# COMMAND ----------

n_trials = 200
n_sim = 3_000
covered_debiased = 0
covered_naive = 0

for trial in range(n_trials):
    rng_t = np.random.default_rng(trial)
    X_t = rng_t.standard_normal((n_sim, p))
    exp_t = rng_t.uniform(0.2, 1.2, size=n_sim)
    eta_t = X_t @ true_coefs + np.log(exp_t)
    y_t = rng_t.poisson(np.exp(eta_t))

    try:
        m_t = PenalizedGLMInference(
            family="poisson", alpha=0.0, l1_ratio=1.0, cv_folds=3, random_state=trial
        ).fit(X_t, y_t, exposure=exp_t)

        df_t = m_t.confidence_intervals(alpha=0.05)
        row = df_t[df_t["feature"] == "age_band"].iloc[0]

        true_val = 0.40
        # Debiased CI
        if row["ci_lower"] <= true_val <= row["ci_upper"]:
            covered_debiased += 1
        # Naive Wald CI around penalized estimate: +/- 1.96 * se
        se_t = float(row["se"])
        pen_t = float(row["coef_penalized"])
        if pen_t - 1.96 * se_t <= true_val <= pen_t + 1.96 * se_t:
            covered_naive += 1

    except Exception:
        pass  # count as miss

print(f"Coverage simulation over {n_trials} trials, n={n_sim}, feature=age_band, true=0.40")
print(f"Debiased CI coverage: {covered_debiased/n_trials*100:.1f}%  (target ~95%)")
print(f"Naive Wald coverage:  {covered_naive/n_trials*100:.1f}%  (expected ~70-80%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC `PenalizedGLMInference` adds a fit-once, explore-both interface to the
# MAGIC insurance-gam toolkit:
# MAGIC
# MAGIC - `fit(X, y, exposure)` — run IRLS + ElasticNet once, store sufficient stats
# MAGIC - `confidence_intervals(alpha)` — asymptotic debiased CIs (Strategy A), fast
# MAGIC - `bootstrap_ci(alpha, n_bootstrap)` — pivot bootstrap CIs (Strategy B), robust
# MAGIC - `summary()` — alias for `confidence_intervals(alpha=0.05)`
# MAGIC - `forest_plot(strategy=...)` — visualise either strategy
# MAGIC
# MAGIC The key insight: the Lasso penalty biases estimates toward zero. For large true
# MAGIC coefficients (NCD, age), this bias is material — naive Wald CIs have ~70-80%
# MAGIC empirical coverage. The one-step Newton correction restores ~95% coverage without
# MAGIC requiring any additional model fits.

print("Demo complete.")
