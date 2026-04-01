"""
Upload insurance-gam post_selection module to Databricks and run tests.
Uses serverless compute via raw REST API (matching other repos in this workspace).

Usage: python notebooks/launch_post_selection_tests.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

api_base = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

RUN_ID = uuid.uuid4().hex[:8]
WORKSPACE_FOLDER = "/Workspace/insurance-gam-ps-tests"
NOTEBOOK_PATH = f"{WORKSPACE_FOLDER}/run_post_selection_tests_{RUN_ID}"

REPO_ROOT = Path(__file__).parent.parent


def api_call(method: str, endpoint: str, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        f"{api_base}/{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"API {method} {endpoint} failed {e.code}: {err_body}")


def read_file(path: Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------------
# Collect files to embed in the notebook
# ---------------------------------------------------------------------------
src_files = {
    "__init__.py": read_file(REPO_ROOT / "src/insurance_gam/__init__.py"),
    "py.typed": "",
    "post_selection.py": read_file(
        REPO_ROOT / "src/insurance_gam/post_selection.py"
    ),
}

test_files = {
    "test_post_selection.py": read_file(
        REPO_ROOT / "tests/test_post_selection.py"
    ),
    "test_helpful_import_errors.py": read_file(
        REPO_ROOT / "tests/test_helpful_import_errors.py"
    ),
}

pyproject_content = read_file(REPO_ROOT / "pyproject.toml")

files_json = json.dumps({"src": src_files, "tests": test_files})
pyproject_json = json.dumps(pyproject_content)

# ---------------------------------------------------------------------------
# Build notebook source
# ---------------------------------------------------------------------------
NOTEBOOK_TEMPLATE = r"""# Databricks notebook source
# MAGIC %pip install statsmodels>=0.14.5 scipy>=1.11.0 scikit-learn>=1.3.0 pandas>=2.0.0 numpy>=2.0 matplotlib>=3.7.0 polars>=1.0 pyarrow>=14.0.0 pytest>=7.4.0 --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
pkg_dir = f"/tmp/insurance_gam_ps_{pkg_id}"
src_dir = f"{pkg_dir}/src/insurance_gam"
tests_dir = f"{pkg_dir}/tests"

os.makedirs(src_dir, exist_ok=True)
os.makedirs(tests_dir, exist_ok=True)

FILES_JSON = __FILES_JSON__
PYPROJECT_JSON = __PYPROJECT_JSON__

data = json.loads(FILES_JSON)
pyproject_src = json.loads(PYPROJECT_JSON)

for name, content in data["src"].items():
    path = f"{src_dir}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)

for name, content in data["tests"].items():
    path = f"{tests_dir}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)

with open(f"{pkg_dir}/pyproject.toml", "w") as fh:
    fh.write(pyproject_src)

print(f"Written {len(data['src'])} source files, {len(data['tests'])} test files to {pkg_dir}")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install STDERR:", r.stderr[:2000])
    raise RuntimeError("pip install failed")
print("insurance-gam installed from", pkg_dir)

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")

env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{tests_dir}/test_post_selection.py",
     f"{tests_dir}/test_helpful_import_errors.py",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider",
    ],
    capture_output=True, text=True,
    cwd=pkg_dir,
    env=env,
)
output = result.stdout + result.stderr
print(output)
print(f"\nReturn code: {result.returncode}")

exit_msg = f"rc={result.returncode}\n" + output[-8000:]
try:
    dbutils.notebook.exit(exit_msg)
except NameError:
    pass
"""

NOTEBOOK_SOURCE = NOTEBOOK_TEMPLATE.replace(
    "__FILES_JSON__", repr(files_json)
).replace("__PYPROJECT_JSON__", repr(pyproject_json))

# ---------------------------------------------------------------------------
# Upload notebook
# ---------------------------------------------------------------------------
print(f"Creating workspace folder {WORKSPACE_FOLDER} ...")
try:
    api_call("POST", "api/2.0/workspace/mkdirs", {"path": WORKSPACE_FOLDER})
    print("Folder ready.")
except RuntimeError as exc:
    print(f"mkdirs note: {exc}")

print(f"Uploading notebook to {NOTEBOOK_PATH} ...")
notebook_b64 = base64.b64encode(NOTEBOOK_SOURCE.encode("utf-8")).decode("ascii")
api_call("POST", "api/2.0/workspace/import", {
    "path": NOTEBOOK_PATH,
    "format": "SOURCE",
    "language": "PYTHON",
    "content": notebook_b64,
    "overwrite": True,
})
print("Notebook uploaded.")

# ---------------------------------------------------------------------------
# Submit job run (serverless — no new_cluster)
# ---------------------------------------------------------------------------
print("Submitting job run ...")
run_resp = api_call("POST", "api/2.1/jobs/runs/submit", {
    "run_name": f"insurance-gam-post-selection-{RUN_ID}",
    "tasks": [{
        "task_key": "pytest",
        "notebook_task": {
            "notebook_path": NOTEBOOK_PATH,
            "source": "WORKSPACE",
        },
    }],
})
run_id = run_resp["run_id"]
print(f"Run ID: {run_id}")
print(f"UI: {api_base}/#job/runs/{run_id}")

# ---------------------------------------------------------------------------
# Poll for completion
# ---------------------------------------------------------------------------
print("Polling for completion (timeout 20 min) ...")
deadline = time.time() + 1200
while time.time() < deadline:
    status = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    life = status["state"]["life_cycle_state"]
    result_state = status["state"].get("result_state", "")
    print(f"  [{time.strftime('%H:%M:%S')}] {life} / {result_state}")
    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(30)

output_resp = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
notebook_out = output_resp.get("notebook_output", {}).get("result", "")
error_msg = output_resp.get("error", "")

print("\n--- Notebook output ---")
if notebook_out:
    print(notebook_out)
if error_msg:
    print("Error:", error_msg)
    trace = output_resp.get("error_trace", "")
    if trace:
        print(trace[:3000])

print(f"\nFinal result: {result_state}")
if result_state == "SUCCESS":
    print("\nRESULT: PASSED")
    sys.exit(0)
else:
    print(f"\nRESULT: FAILED ({result_state})")
    sys.exit(1)
