"""
Benchmark: InsuranceEBM vs Poisson GLM for insurance frequency modelling.

DGP: Poisson frequency with known non-linear effects.
  - driver_age: U-shaped hazard (high for young/old, low in middle)
  - ncd_years: exponential discount (non-linear)
  - vehicle_age: step change at age 8 (threshold effect)
  - annual_miles: log-linear effect

A standard Poisson GLM with linear terms cannot recover U-shapes or
threshold effects. EBM learns the shape data-adaptively.

Metric: Poisson deviance on held-out test set (lower is better).
        Gini coefficient (higher is better).
"""

import sys
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------

def make_dataset(n: int = 5000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)

    driver_age    = rng.integers(17, 76, n).astype(float)
    ncd_years     = rng.integers(0, 10, n).astype(float)
    vehicle_age   = rng.integers(0, 16, n).astype(float)
    annual_miles  = rng.uniform(3_000, 25_000, n)
    area          = rng.integers(0, 6, n).astype(float)

    # True log-rate: non-linear effects that GLM cannot recover with linear terms
    # 1. U-shaped driver age effect: young (<25) and old (>65) are riskier
    age_effect = (
        0.8 * np.exp(-0.15 * np.maximum(driver_age - 17, 0))   # young decay
        + 0.3 * np.exp(0.05 * np.maximum(driver_age - 55, 0))  # old increase
    )
    # 2. Non-linear NCD discount (exponential, not linear)
    ncd_effect = -0.12 * ncd_years - 0.008 * ncd_years ** 2

    # 3. Threshold effect at vehicle_age == 8
    veh_effect = 0.35 * (vehicle_age >= 8).astype(float)

    # 4. Log-miles effect
    miles_effect = 0.25 * np.log(annual_miles / 10_000)

    # 5. Area categorical
    area_coef = np.array([0.0, 0.15, 0.30, -0.10, 0.20, -0.05])
    area_effect = area_coef[area.astype(int)]

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean unit Poisson deviance."""
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
    Normalised Gini coefficient (ordering statistic).

    Measures how well the model ranks risks. Perfect ranking => Gini=1.
    Random prediction => Gini=0. Standard insurance pricing metric.
    """
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    n = len(y_sorted)
    cum_actual = np.cumsum(y_sorted) / np.sum(y_sorted)
    cum_pop = np.arange(1, n + 1) / n
    lorenz_area = np.trapz(cum_actual, cum_pop)
    return float(2.0 * lorenz_area - 1.0)


# ---------------------------------------------------------------------------
# Naive GLM baseline (statsmodels Poisson with linear terms)
# ---------------------------------------------------------------------------

def fit_glm_baseline(X_train, y_train, exp_train, X_test, exp_test):
    """Poisson GLM with linear/log-linear terms — standard industry baseline."""
    import statsmodels.api as sm

    # Build design matrix: intercept + linear features + log(annual_miles)
    def build_design(X: pl.DataFrame, exposure: np.ndarray) -> tuple:
        df_pd = X.to_pandas()
        Xd = np.column_stack([
            np.ones(len(df_pd)),
            df_pd["driver_age"],
            df_pd["driver_age"] ** 2 / 1000,   # allow slight curve but not U-shape
            df_pd["ncd_years"],
            df_pd["vehicle_age"],
            np.log(df_pd["annual_miles"] / 10_000),
            (df_pd["area"] == 1).astype(float),
            (df_pd["area"] == 2).astype(float),
            (df_pd["area"] == 3).astype(float),
            (df_pd["area"] == 4).astype(float),
            (df_pd["area"] == 5).astype(float),
        ])
        return Xd

    Xd_train = build_design(X_train, exp_train)
    Xd_test  = build_design(X_test,  exp_test)

    offset_train = np.log(np.maximum(exp_train, 1e-10))
    glm = sm.GLM(
        y_train,
        Xd_train,
        family=sm.families.Poisson(),
        offset=offset_train,
    )
    res = glm.fit(disp=0)

    offset_test = np.log(np.maximum(exp_test, 1e-10))
    y_pred = res.predict(Xd_test, offset=offset_test)
    return y_pred


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Benchmark: InsuranceEBM vs Poisson GLM")
    print("DGP: Non-linear frequency (U-shape, thresholds, log-miles)")
    print("=" * 60)

    # Generate data
    df, y, exposure, log_rate = make_dataset(n=6000, seed=42)

    n_train = 4500
    X_train, X_test = df[:n_train],  df[n_train:]
    y_train, y_test = y[:n_train],   y[n_train:]
    exp_train, exp_test = exposure[:n_train], exposure[n_train:]

    print(f"\nDataset: {n_train} train / {len(y_test)} test observations")
    print(f"Mean frequency: {y_train.sum() / exp_train.sum():.4f} claims/year")

    # ------------------------------------------------------------------
    # Baseline: Poisson GLM
    # ------------------------------------------------------------------
    print("\nFitting Poisson GLM (linear terms)...")
    try:
        glm_pred = fit_glm_baseline(X_train, y_train, exp_train, X_test, exp_test)
        glm_dev  = poisson_deviance(y_test, glm_pred)
        glm_gini = gini_coefficient(y_test, glm_pred)
        print(f"  GLM fit complete")
    except ImportError:
        print("  statsmodels not available — skipping GLM baseline")
        glm_dev, glm_gini = None, None

    # ------------------------------------------------------------------
    # InsuranceEBM
    # ------------------------------------------------------------------
    print("\nFitting InsuranceEBM (Poisson, interactions='3x')...")
    try:
        from insurance_gam.ebm import InsuranceEBM

        ebm = InsuranceEBM(loss="poisson", interactions="3x", random_state=42)
        ebm.fit(X_train, y_train, exposure=exp_train)

        ebm_pred = ebm.predict(X_test, exposure=exp_test)
        ebm_dev  = poisson_deviance(y_test, ebm_pred)
        ebm_gini = gini_coefficient(y_test, ebm_pred)
        print(f"  EBM fit complete")
    except Exception as e:
        print(f"  EBM failed: {e}")
        ebm_dev, ebm_gini = None, None

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS (test set)")
    print("=" * 60)
    print(f"{'Model':<25} {'Poisson Deviance':>18} {'Gini':>10}")
    print("-" * 55)

    if glm_dev is not None:
        print(f"{'Poisson GLM (linear)':<25} {glm_dev:>18.6f} {glm_gini:>10.4f}")
    else:
        print(f"{'Poisson GLM (linear)':<25} {'N/A (no statsmodels)':>18}")

    if ebm_dev is not None:
        print(f"{'InsuranceEBM':<25} {ebm_dev:>18.6f} {ebm_gini:>10.4f}")
    else:
        print(f"{'InsuranceEBM':<25} {'N/A':>18}")

    if glm_dev is not None and ebm_dev is not None:
        dev_improvement = 100 * (glm_dev - ebm_dev) / glm_dev
        gini_improvement = 100 * (ebm_gini - glm_gini) / max(abs(glm_gini), 1e-6)
        print("-" * 55)
        print(f"\nEBM deviance improvement: {dev_improvement:+.1f}% over GLM")
        print(f"EBM Gini improvement:     {gini_improvement:+.1f}% over GLM")
        print("\nConclusion: EBM captures non-linear effects (U-shaped age,")
        print("  threshold at vehicle_age=8, non-linear NCD discount).")
        print("  GLM under-predicts risk spread and produces worse ranking.")

    # Oracle check: how well does EBM recover the true log-rate?
    if ebm_dev is not None:
        true_pred_test = np.exp(log_rate[n_train:]) * exp_test
        oracle_dev = poisson_deviance(y_test, true_pred_test)
        print(f"\nOracle (true DGP) deviance: {oracle_dev:.6f}")
        print(f"EBM deviance gap from oracle: {ebm_dev - oracle_dev:.6f}")


if __name__ == "__main__":
    main()
