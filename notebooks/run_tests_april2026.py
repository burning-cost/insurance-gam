# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-gam: full test suite including April 2026 coverage expansion
# MAGIC
# MAGIC Runs all GLM inference tests:
# MAGIC   - test_post_selection.py
# MAGIC   - test_debiased_glm.py
# MAGIC   - test_penalized_glm_inference.py
# MAGIC   - test_new_coverage_20250403.py
# MAGIC   - test_coverage_april2026.py (new)

# COMMAND ----------

# MAGIC %pip install statsmodels>=0.14.5 scipy>=1.11.0 scikit-learn>=1.3.0 pandas>=2.0.0 numpy>=2.0 matplotlib>=3.7.0 polars>=1.0 pyarrow>=14.0.0 pytest>=7.4.0

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
     "tests/test_post_selection.py",
     "tests/test_debiased_glm.py",
     "tests/test_penalized_glm_inference.py",
     "tests/test_new_coverage_20250403.py",
     "tests/test_coverage_april2026.py",
     "tests/test_helpful_import_errors.py",
     "-v", "--tb=short", "--no-header", "--color=no",
    ],
    capture_output=True, text=True,
    cwd=dst,
    env=env,
)
output = result.stdout + result.stderr
print(output[-40000:] if len(output) > 40000 else output)

dbutils.notebook.exit(output[-20000:] if len(output) > 20000 else output)
