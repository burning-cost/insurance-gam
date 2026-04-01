# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-gam DebiasedGLM test runner
# MAGIC
# MAGIC Runs the full test suite including the new DebiasedGLM module.
# MAGIC Tests: DebiasedGLM (Poisson, Gamma, Tweedie), bootstrap Strategy B,
# MAGIC exposure offsets, forest_plot, input validation, and all pre-existing tests.

# COMMAND ----------

# MAGIC %pip install statsmodels>=0.14.5 scipy>=1.11.0 scikit-learn>=1.3.0 pandas>=2.0.0 numpy>=2.0 matplotlib>=3.7.0 polars>=1.0 pyarrow>=14.0.0 pytest>=7.4.0 pytest-cov>=4.0.0

# COMMAND ----------

import subprocess, sys, os, shutil, uuid

run_id = str(uuid.uuid4())[:8]
src = "/Workspace/insurance-gam"
dst = f"/tmp/insurance-gam-{run_id}"

SKIP = {".venv", ".git", "__pycache__", ".pytest_cache", "dist"}

def _ignore(directory, contents):
    ignored = set()
    for item in contents:
        if item in SKIP or item.endswith(".egg-info") or item.endswith(".pyc"):
            ignored.add(item)
    return ignored

shutil.copytree(src, dst, ignore=_ignore)
print(f"Copied {src} -> {dst}")

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", dst],
    capture_output=True, text=True, cwd=dst,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("pip install failed")
print("Install OK.")

# COMMAND ----------

import subprocess, sys, os

env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_debiased_glm.py",
     "tests/test_post_selection.py",
     "tests/test_helpful_import_errors.py",
     "-v", "--tb=short", "--no-header", "--color=no",
    ],
    capture_output=True, text=True,
    cwd=dst,
    env=env,
)
output = result.stdout + result.stderr
print(output[-30000:] if len(output) > 30000 else output)

dbutils.notebook.exit(output[-20000:] if len(output) > 20000 else output)
