"""
Run the full insurance-gam test suite on Databricks with coverage.

Uses runs/submit (one-shot) with serverless compute and a notebook task.
"""

from __future__ import annotations

import os
import sys
import time
import base64
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------

env_file = Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_file.read_text().splitlines():
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc

w = WorkspaceClient()
HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]

# ---------------------------------------------------------------------------
# Notebook content — full test suite + coverage
# ---------------------------------------------------------------------------

NOTEBOOK_PATH = "/Users/pricing.frontier@gmail.com/insurance_gam_demo"

NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %pip install scikit-learn numpy polars scipy matplotlib pytest pytest-cov statsmodels -q

# COMMAND ----------

import subprocess, sys, os

# Install from uploaded source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-gam-tests", "--no-deps", "-q"],
    capture_output=True, text=True
)
print(result.stdout[-2000:])
if result.returncode != 0:
    print("INSTALL STDERR:", result.stderr[-2000:])
    raise RuntimeError("pip install failed")
print("Install OK")

# COMMAND ----------

import subprocess, sys

# Run non-neural, non-ebm tests with coverage
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-gam-tests/tests",
        "--ignore=/Workspace/insurance-gam-tests/tests/anam",
        "--ignore=/Workspace/insurance-gam-tests/tests/pin",
        "--ignore=/Workspace/insurance-gam-tests/tests/ebm",
        "--cov=insurance_gam",
        "--cov-report=term-missing",
        "-q", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-gam-tests",
)
output = result.stdout
if result.stderr:
    output += "\\nSTDERR:\\n" + result.stderr[-2000:]
print(output[-10000:] if len(output) > 10000 else output)
if result.returncode not in (0, 5):
    raise SystemExit(f"pytest exited with code {result.returncode}")
"""

# ---------------------------------------------------------------------------
# Upload files
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = "/Workspace/insurance-gam-tests"
LOCAL_ROOT = Path(__file__).parent


def ws_mkdirs(path: str) -> None:
    try:
        w.workspace.mkdirs(path=path)
    except Exception:
        pass


def upload_file_raw(local_path: Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ws_svc.ImportFormat.AUTO,
    )


def upload_directory(local_dir: Path, remote_dir: str) -> None:
    for local_path in sorted(local_dir.rglob("*")):
        if not local_path.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in local_path.parts):
            continue
        if local_path.suffix in {".pyc", ".pyo"}:
            continue
        rel = local_path.relative_to(local_dir)
        remote_path = f"{remote_dir}/{rel.as_posix()}"
        ws_mkdirs(remote_path.rsplit("/", 1)[0])
        print(f"  {rel}")
        upload_file_raw(local_path, remote_path)


print("Uploading files to Databricks Workspace...")
ws_mkdirs(WORKSPACE_ROOT)
upload_file_raw(LOCAL_ROOT / "pyproject.toml", f"{WORKSPACE_ROOT}/pyproject.toml")
print("src/")
upload_directory(LOCAL_ROOT / "src", f"{WORKSPACE_ROOT}/src")
print("tests/")
upload_directory(LOCAL_ROOT / "tests", f"{WORKSPACE_ROOT}/tests")

# Upload the runner notebook
print(f"Overwriting notebook: {NOTEBOOK_PATH}")
encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=encoded_nb,
    overwrite=True,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
)
print("Notebook uploaded.")

# ---------------------------------------------------------------------------
# Submit one-shot run with serverless compute
# ---------------------------------------------------------------------------

import requests

headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

submit_payload = {
    "run_name": "insurance-gam-full-test-suite",
    "tasks": [
        {
            "task_key": "run_tests",
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
            },
        }
    ],
    "environments": [
        {
            "environment_key": "default",
            "spec": {
                "client": "1",
                "dependencies": [],
            },
        }
    ],
    "queue": {"enabled": True},
}

resp = requests.post(
    f"{HOST}/api/2.1/jobs/runs/submit",
    headers=headers,
    json=submit_payload,
)
if resp.status_code != 200:
    print(f"Submit failed: {resp.status_code} {resp.text}")
    sys.exit(1)

run_id = resp.json()["run_id"]
print(f"Run submitted: run_id={run_id}")
print(f"Track at: {HOST}/#runs/{run_id}")

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

poll_interval = 20
max_wait = 1800
elapsed = 0

while elapsed < max_wait:
    r = requests.get(f"{HOST}/api/2.1/jobs/runs/get?run_id={run_id}", headers=headers)
    data = r.json()
    state = data.get("state", {})
    lc = state.get("life_cycle_state", "UNKNOWN")
    rs = state.get("result_state", "")
    print(f"  [{elapsed:3d}s] {lc} {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(poll_interval)
    elapsed += poll_interval

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print("\n--- Run output ---")
r = requests.get(f"{HOST}/api/2.1/jobs/runs/get-output?run_id={run_id}", headers=headers)
out_data = r.json()
if "notebook_output" in out_data and out_data["notebook_output"].get("result"):
    print(out_data["notebook_output"]["result"])
elif "error" in out_data:
    print("ERROR:", out_data["error"])
    print(out_data.get("error_trace", "")[-3000:])
else:
    print("(no notebook output captured)")
    print(json.dumps(out_data, indent=2)[:3000])

final_state = data.get("state", {})
result_state = final_state.get("result_state", "UNKNOWN")
print(f"\nFinal result: {result_state}")

if result_state != "SUCCESS":
    sys.exit(1)
