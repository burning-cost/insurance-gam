# Databricks notebook source
# MAGIC %md
# MAGIC # DebiasedGLM: Bias-corrected CIs for Lasso-selected GLM coefficients
# MAGIC
# MAGIC **Problem**: After Lasso selects rating factors for a Poisson frequency or
# MAGIC Gamma severity model, standard Wald confidence intervals on those coefficients
# MAGIC are **invalid**. The Lasso shrinkage biases estimates toward zero, and conditioning
# MAGIC on the selection event makes naive CIs too narrow.
# MAGIC
# MAGIC **Solution**: The debiased Lasso estimator (Manna et al. 2025, arXiv:2410.01008)
# MAGIC applies a one-step Newton correction using the inverse Hessian of the GLM
# MAGIC log-likelihood. The result is an asymptotically valid Wald-style CI — valid
# MAGIC unconditionally (marginally), not conditioned on any particular selection event.
# MAGIC
# MAGIC **When to use**:
# MAGIC - Poisson frequency models after elastic net selection
# MAGIC - Gamma severity models (PostSelectionGLM does not support Gamma)
# MAGIC - Tweedie combined models
# MAGIC - Reporting "what is the plausible range for this rate factor?" rather than
# MAGIC   "is this factor genuinely significant?" (use PostSelectionGLM for the latter)
# MAGIC
# MAGIC **Validity condition**: p^2/n -> 0. At n=100,000 and p=50 selected factors,
# MAGIC p^2/n = 0.025 — well-satisfied for typical UK motor portfolios.

# COMMAND ----------

# MAGIC %pip install insurance-gam[glm]

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_gam.debiased_glm import DebiasedGLM

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic Poisson frequency model
# MAGIC
# MAGIC Simulate a portfolio with 10 candidate rating factors, 3 of which are truly
# MAGIC predictive (vehicle group proxy, driver age proxy, NCB proxy).
# MAGIC Log-relativity magnitudes: [0.4, 0.3, 0.2] — realistic for motor frequency.

# COMMAND ----------

rng = np.random.default_rng(42)
n, p = 5000, 10
true_coefs = np.array([0.4, 0.3, 0.2])  # x0, x1, x2

X = rng.standard_normal((n, p))
exposure = rng.uniform(0.5, 2.0, size=n)  # policy years
eta = X[:, :3] @ true_coefs + np.log(exposure)
y_freq = rng.poisson(np.exp(eta))

print(f"Dataset: n={n}, p={p}, mean claims per policy-year = {y_freq.sum()/exposure.sum():.3f}")
print(f"True coefs on x0/x1/x2: {true_coefs}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Strategy A: Asymptotic debiased estimator (fast, good for large portfolios)

# COMMAND ----------

model_a = DebiasedGLM(
    family="poisson",
    alpha=0.0,          # 0.0 = use CV to select lambda
    l1_ratio=0.5,       # elastic net (handles correlated factors)
    confidence=0.95,
    n_bootstrap=0,      # Strategy A
    random_state=42,
)
model_a.fit(X, y_freq, exposure=exposure)

df_a = model_a.summary()
print(f"Strategy A: lambda={model_a.lambda_:.4f}, selected={model_a.selected_features_.sum()} features")
print()
print(df_a.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Check CI coverage on true features

# COMMAND ----------

for name, true_val in [("x0", 0.4), ("x1", 0.3), ("x2", 0.2)]:
    row = df_a[df_a["feature"] == name].iloc[0]
    covered = row["ci_lower"] <= true_val <= row["ci_upper"]
    print(
        f"{name}: true={true_val:.2f}  coef={row['coef']:.4f}  "
        f"95% CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]  "
        f"selected={row['selected']}  covered={covered}"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ### Forest plot

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 5))
model_a.forest_plot(ax=ax)
ax.set_title("Debiased GLM — Poisson frequency model (Strategy A)")
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Strategy B: Pearson residual bootstrap (better for smaller n or uncertain sparsity)

# COMMAND ----------

model_b = DebiasedGLM(
    family="poisson",
    alpha=model_a.lambda_,  # fix lambda from Strategy A for fair comparison
    l1_ratio=0.5,
    confidence=0.95,
    n_bootstrap=200,         # Strategy B: 200 bootstrap resamples
    random_state=42,
)
model_b.fit(X, y_freq, exposure=exposure)

df_b = model_b.summary()
print("Strategy B (bootstrap, n_bootstrap=200):")
print(df_b[df_b["selected"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Gamma severity model
# MAGIC
# MAGIC Gamma fills the gap left by PostSelectionGLM (Poisson only).
# MAGIC Severity model: log-linear mean, shape=3, no exposure.

# COMMAND ----------

shape = 3.0
mu_sev = np.exp(X[:, :3] @ true_coefs)
y_sev = rng.gamma(shape, mu_sev / shape)

model_gamma = DebiasedGLM(
    family="gamma",
    alpha=0.0,
    l1_ratio=0.5,
    confidence=0.95,
    random_state=42,
)
model_gamma.fit(X, y_sev)

df_gamma = model_gamma.summary()
print(f"Gamma model: phi={model_gamma.phi_:.4f} (true phi=1/shape={1/shape:.4f})")
print()
print(df_gamma.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 5))
model_gamma.forest_plot(ax=ax)
ax.set_title("Debiased GLM — Gamma severity model (Strategy A)")
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Tweedie combined model (power=1.5)

# COMMAND ----------

phi_true = 2.0
mu_tw = np.exp(X[:, :3] @ true_coefs)
# Approximate Tweedie using Gamma as proxy for demo
y_tw = rng.gamma(mu_tw / phi_true, phi_true)
y_tw = np.clip(y_tw, 1e-6, None)

model_tw = DebiasedGLM(
    family="tweedie",
    tweedie_power=1.5,
    alpha=0.0,
    l1_ratio=0.5,
    confidence=0.95,
    random_state=42,
)
model_tw.fit(X, y_tw)

df_tw = model_tw.summary()
print(f"Tweedie model: phi={model_tw.phi_:.4f}, lambda={model_tw.lambda_:.4f}")
print()
print(df_tw[df_tw["selected"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Combined workflow: PostSelectionGLM + DebiasedGLM
# MAGIC
# MAGIC Use both in sequence:
# MAGIC - PostSelectionGLM: "Is vehicle group genuinely significant given Lasso selected it?"
# MAGIC - DebiasedGLM: "What is the plausible range for the vehicle group log-relativity?"

# COMMAND ----------

from insurance_gam.post_selection import PostSelectionGLM

psi = PostSelectionGLM(family="poisson", alpha=0.05, random_state=42).fit(
    X[:, :3],  # use only selected features for illustration
    y_freq,
    exposure=exposure,
)
df_psi = psi.summary()

deb = DebiasedGLM(family="poisson", alpha=0.1, confidence=0.95, random_state=42).fit(
    X[:, :3],
    y_freq,
    exposure=exposure,
)
df_deb = deb.summary()

comparison = pd.merge(
    df_psi[["feature", "coefficient", "ci_lower", "ci_upper", "pvalue"]].rename(
        columns={"coefficient": "psi_coef", "ci_lower": "psi_lo", "ci_upper": "psi_hi", "pvalue": "psi_pval"}
    ),
    df_deb[["feature", "coef", "ci_lower", "ci_upper", "pvalue"]].rename(
        columns={"coef": "deb_coef", "ci_lower": "deb_lo", "ci_upper": "deb_hi", "pvalue": "deb_pval"}
    ),
    on="feature",
)
print("Comparison: PSI (conditional) vs Debiased (marginal)")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Method | Family | CI type | Speed | Use case |
# MAGIC |--------|--------|---------|-------|----------|
# MAGIC | PostSelectionGLM | Poisson | Conditional (truncated normal) | Slow (path tracing) | Is this factor genuinely non-zero? |
# MAGIC | DataSplitPostSelectionGLM | Poisson | Marginal Wald (split-data) | Fast | Simple valid CIs, large n |
# MAGIC | DebiasedGLM (Strategy A) | Poisson, Gamma, Tweedie | Marginal debiased Wald | Fast O(np^2) | Reporting rate factor magnitudes |
# MAGIC | DebiasedGLM (Strategy B) | Poisson, Gamma, Tweedie | Bootstrap pivot | Slow (B Lasso fits) | Small n or weak sparsity |

print("Demo complete.")
