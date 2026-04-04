"""
Run the insurance-gam Boulevard test suite on Databricks.

Strategy:
1. Upload src/ and tests/ to /Workspace/insurance-gam-tests/
2. Overwrite the existing insurance_gam_demo notebook in the user's workspace
   with a pytest runner (the validate-insurance-gam job points at it)
3. Trigger that job via run-now and poll for completion

This approach works around the one-shot submit endpoint being temporarily
unavailable — run-now against an existing job uses the same infrastructure.
"""

from __future__ import annotations

import os
import sys
import time
import base64
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
from databricks.sdk.service import workspace as ws_svc, jobs, compute

w = WorkspaceClient()
HOST = os.environ["DATABRICKS_HOST"].rstrip("/")

# ---------------------------------------------------------------------------
# Notebook content — uploaded to the existing job's notebook path
# ---------------------------------------------------------------------------

# The validate-insurance-gam job (ID 896318314048217) points at this notebook.
NOTEBOOK_PATH = "/Users/pricing.frontier@gmail.com/insurance_gam_demo"
VALIDATE_JOB_ID = 896318314048217

NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %pip install scikit-learn numpy polars scipy matplotlib pytest -q

# COMMAND ----------

import subprocess, sys, os

# Install from uploaded source (no-deps: all deps already installed above)
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-gam-tests", "--no-deps", "-q"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("INSTALL STDERR:", result.stderr[-2000:])
    raise RuntimeError("pip install failed")
print("Install OK")

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-gam-tests/tests/ebm/test_boulevard.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-gam-tests",
)
output = result.stdout
if result.stderr:
    output += "\\nSTDERR:\\n" + result.stderr[-1000:]
# Databricks notebook output is capped; print the tail which has the summary
print(output[-8000:] if len(output) > 8000 else output)
if result.returncode != 0:
    raise SystemExit(f"pytest exited with code {result.returncode}")
"""

# ---------------------------------------------------------------------------
# Upload source and test files
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

# Upload the runner notebook over the existing demo notebook
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
# Trigger the existing validate-insurance-gam job
# ---------------------------------------------------------------------------

print(f"\nTriggering job {VALIDATE_JOB_ID} (validate-insurance-gam)...")
run_response = w.jobs.run_now(job_id=VALIDATE_JOB_ID)
run_id = run_response.run_id
print(f"Run submitted: run_id={run_id}")
print(f"Track at: {HOST}/#job/{VALIDATE_JOB_ID}/run/{run_id}")

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

poll_interval = 20
max_wait = 900
elapsed = 0

while elapsed < max_wait:
    status = w.jobs.get_run(run_id=run_id)
    state = status.state
    lc = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    rs = state.result_state.value if state.result_state else ""
    print(f"  [{elapsed:3d}s] {lc} {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(poll_interval)
    elapsed += poll_interval

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print("\n--- Run output ---")
try:
    output = w.jobs.get_run_output(run_id=run_id)
    if output.notebook_output and output.notebook_output.result:
        print(output.notebook_output.result)
    elif output.error:
        print("ERROR:", output.error)
        if output.error_trace:
            print(output.error_trace[-3000:])
    else:
        print("(no notebook output captured)")
except Exception as exc:
    print(f"Could not fetch output: {exc}")

final = w.jobs.get_run(run_id=run_id).state
result = final.result_state.value if final.result_state else "UNKNOWN"
print(f"\nFinal result: {result}")

if result != "SUCCESS":
    sys.exit(1)
