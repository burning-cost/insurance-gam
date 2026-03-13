# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install torch>=2.0.0 polars>=0.20.0 numpy>=1.24.0 matplotlib>=3.7.0 scikit-learn>=1.3.0 pandas>=2.0.0 pyarrow>=10.0.0 pytest>=7.4.0

# COMMAND ----------

import subprocess, sys, os, shutil

# Copy project to /tmp (writable) to avoid __pycache__ issues on workspace FS
src = "/Workspace/insurance-gam"
dst = "/tmp/insurance-gam"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("Copied to", dst)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", dst],
    capture_output=True, text=True, cwd=dst,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
)
print(result.stdout[-1000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
    raise RuntimeError("pip install failed")

# COMMAND ----------

import subprocess, sys, os

env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/ebm/", "tests/anam/", "tests/pin/",
     "-v", "--tb=long", "--no-header",
    ],
    capture_output=True, text=True, cwd="/tmp/insurance-gam/",
    env=env,
)
output = result.stdout + result.stderr
print(output[-12000:] if len(output) > 12000 else output)
print(f"\nReturn code: {result.returncode}")
dbutils.notebook.exit(f"rc={result.returncode}\n" + output[-4000:])
