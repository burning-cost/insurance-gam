"""
Run insurance-gam integration tests on Databricks.
Tests: tests/test_integration.py
Uses existing job: validate-insurance-gam (896318314048217)
by overwriting /Users/pricing.frontier@gmail.com/insurance_gam_demo
"""
from __future__ import annotations

import os
import sys
import time
import base64
from pathlib import Path

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

WORKSPACE_ROOT = "/Workspace/insurance-gam-tests"
NOTEBOOK_PATH = "/Users/pricing.frontier@gmail.com/insurance_gam_demo"
JOB_ID = 896318314048217

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
        if any(part.startswith(".") or part in ("__pycache__", "dist", "build") for part in local_path.parts):
            continue
        if local_path.suffix in {".pyc", ".pyo"}:
            continue
        rel = local_path.relative_to(local_dir)
        remote_path = f"{remote_dir}/{rel.as_posix()}"
        ws_mkdirs(remote_path.rsplit("/", 1)[0])
        print(f"  {rel}")
        upload_file_raw(local_path, remote_path)


print("Uploading insurance-gam source + tests...")
ws_mkdirs(WORKSPACE_ROOT)
upload_file_raw(LOCAL_ROOT / "pyproject.toml", f"{WORKSPACE_ROOT}/pyproject.toml")
upload_directory(LOCAL_ROOT / "src", f"{WORKSPACE_ROOT}/src")
upload_directory(LOCAL_ROOT / "tests", f"{WORKSPACE_ROOT}/tests")

# Install torch CPU-only in the notebook. Databricks clusters may or may not have it.
NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %pip install scikit-learn numpy polars scipy matplotlib pytest -q

# COMMAND ----------

import subprocess, sys, os, shutil

log = []

# Install torch CPU-only (small wheel, avoids CUDA size issues)
r_torch = subprocess.run(
    [sys.executable, "-m", "pip", "install", "torch",
     "--index-url", "https://download.pytorch.org/whl/cpu", "-q"],
    capture_output=True, text=True
)
log.append(f"TORCH_RC={r_torch.returncode}")
if r_torch.returncode != 0:
    log.append("TORCH_ERR:" + r_torch.stderr[-500:])
    dbutils.notebook.exit("\\n".join(log))

# Install from uploaded source
r_inst = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-gam-tests", "--no-deps", "-q"],
    capture_output=True, text=True
)
log.append(f"INSTALL_RC={r_inst.returncode}")
if r_inst.returncode != 0:
    log.append("INSTALL_ERR:" + r_inst.stderr[-500:])
    dbutils.notebook.exit("\\n".join(log))
log.append("Install OK")

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-gam-tests/tests/test_integration.py",
        "-v", "--tb=short", "--no-header",
        "-p", "no:cacheprovider",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-gam-tests",
)
output = result.stdout
if result.stderr:
    output += "\\nSTDERR:\\n" + result.stderr[-1000:]
print(output[-12000:] if len(output) > 12000 else output)
log.append(f"PYTEST_RC={result.returncode}")
log.append(output[-8000:] if len(output) > 8000 else output)
dbutils.notebook.exit("\\n".join(log))
"""

print(f"Uploading runner notebook to {NOTEBOOK_PATH}")
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=base64.b64encode(NOTEBOOK_CONTENT.encode()).decode(),
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print("Notebook uploaded.")

print(f"\nTriggering job {JOB_ID} (validate-insurance-gam)...")
run_response = w.jobs.run_now(job_id=JOB_ID)
run_id = run_response.run_id
print(f"Run submitted: run_id={run_id}")
print(f"Track at: {HOST}/#job/{JOB_ID}/run/{run_id}")

poll_interval = 20
max_wait = 1200
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

print("\n--- Run output ---")
nb_result = ""
try:
    output = w.jobs.get_run_output(run_id=run_id)
    if output.notebook_output and output.notebook_output.result:
        nb_result = output.notebook_output.result
        print(nb_result)
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
