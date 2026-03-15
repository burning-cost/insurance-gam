# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-gam test runner

# COMMAND ----------

# MAGIC %pip install torch>=2.0.0 polars>=0.20.0 numpy>=1.24.0 matplotlib>=3.7.0 scikit-learn>=1.3.0 pandas>=2.0.0 pyarrow>=10.0.0 pytest>=7.4.0 pytest-cov>=4.0.0

# COMMAND ----------

import subprocess, sys, os, shutil, uuid

run_id = str(uuid.uuid4())[:8]
src = "/Workspace/insurance-gam"
dst = f"/tmp/insurance-gam-{run_id}"

SKIP = {".venv", ".git", "__pycache__", ".pytest_cache"}

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
    print("STDERR:", result.stderr[-500:])
    raise RuntimeError("pip install failed")
print("Install OK.")

# COMMAND ----------

import subprocess, sys, os

env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/ebm/", "tests/anam/", "tests/pin/",
     "-v", "--tb=short", "--no-header", "--color=no",
    ],
    capture_output=True, text=True,
    cwd=dst,
    env=env,
)
output = result.stdout + result.stderr

# Use dbutils.notebook.exit to return the output as the notebook result
# This is the correct way to pass output back through the Jobs API
dbutils.notebook.exit(output[-20000:] if len(output) > 20000 else output)
