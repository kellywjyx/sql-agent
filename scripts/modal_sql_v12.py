"""Run the V12 arms on Modal GPUs with Ollama and the unchanged sql_agent pipeline in one container.

The Ollama client accepts only localhost, so generation, read-only SQL execution, and scoring all run
inside the container. Usage from the sql-agent directory:

    modal run scripts/modal_sql_v12.py --stage setup         # upload dbs/data, pull model, build value profiles
    modal run scripts/modal_sql_v12.py --stage smoke         # 6 development cases, control + full
    modal run scripts/modal_sql_v12.py --stage development   # all four arms, 200 cases each
    modal run scripts/modal_sql_v12.py --stage pull          # then freeze_development locally
    modal run scripts/modal_sql_v12.py --stage final         # control + frozen winner, 1,033 cases
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parent.parent
LOCAL = PROJECT.parent / "artifacts"
WHEEL = LOCAL / "releases/v0.4.0/wheels/portfolio_llm_evals-0.4.0-py3-none-any.whl"
REMOTE = Path("/artifacts")
MODEL = "qwen2.5-coder:7b-instruct"
GPU = "A10G"
USD_PER_GPU_SECOND = 1.10 / 3600  # estimate for the running cap; Modal billing is authoritative
COST_CAP_USD = 15.0
LEDGER = LOCAL / "v12/compute/modal-ledger.json"
DATASETS = ["v6/sql-agent/data/development.jsonl", "v6/sql-agent/data/final.jsonl"]

volume = modal.Volume.from_name("sql-v12-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("pydantic==2.12.5", "sqlglot==30.0.1", "numpy==2.5.2", "httpx")
    .add_local_file(str(WHEEL), f"/wheels/{WHEEL.name}", copy=True)
    .run_commands(f"pip install /wheels/{WHEEL.name}")
    .env({"PYTHONPATH": "/src", "OLLAMA_MODELS": "/artifacts/ollama", "OLLAMA_HOST": "127.0.0.1:11434",
          "OLLAMA_NUM_PARALLEL": "1"})
    .add_local_dir(str(PROJECT / "src/sql_agent"), "/src/sql_agent", ignore=["__pycache__"])
)
app = modal.App("sql-agent-v12", image=image)


def _serve():
    import subprocess
    import httpx
    process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            httpx.get("http://127.0.0.1:11434/api/tags", timeout=2).raise_for_status()
            return process
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("Ollama did not start")


def _link_databases():
    """Expose uploaded databases under the frozen Windows path strings without rewriting any dataset."""
    import os
    work = Path("/work")
    work.mkdir(exist_ok=True)
    os.chdir(work)
    for dataset in DATASETS:
        for line in (REMOTE / dataset).read_text(encoding="utf-8").splitlines():
            frozen = json.loads(line)["context"]["database"]
            link = work / frozen
            if not link.exists():
                stem = frozen.replace("\\", "/").split("/")[-1].removesuffix(".sqlite")
                link.symlink_to(REMOTE / "dbs" / stem / f"{stem}.sqlite")


@app.function(volumes={str(REMOTE): volume}, timeout=3600, cpu=4, memory=16384)
def prepare() -> dict:
    import subprocess
    import httpx
    from sql_agent.schema import inspect_schema
    from sql_agent.value_profiles import ValueProfiler
    volume.reload()
    _serve()
    subprocess.run(["ollama", "pull", MODEL], check=True, capture_output=True)
    digest = next(m["digest"] for m in httpx.get("http://127.0.0.1:11434/api/tags").json()["models"] if m["name"] == MODEL)
    profiler, built = ValueProfiler(REMOTE), []
    for folder in sorted((REMOTE / "dbs").iterdir()):
        database = folder / f"{folder.name}.sqlite"
        built.append({"database": folder.name, "identity": profiler.build(database, inspect_schema(database))["identity"]})
    volume.commit()
    return {"model": MODEL, "digest": digest, "value_profiles": built}


@app.function(volumes={str(REMOTE): volume}, gpu=GPU, timeout=5 * 3600, cpu=4, memory=32768)
def run_arm(stage: str, arm: str) -> dict:
    from sql_agent import v12_campaign
    volume.reload()
    _serve()
    _link_databases()
    started = time.time()
    try:
        report = v12_campaign.run(REMOTE, stage, arm)
        result = {"run_id": report["run_id"], "metrics": report["metrics"], "p95_seconds": report["p95_seconds"],
                  "failed": report.get("failed"), "status": report.get("status"),
                  "database_hash_mismatches": report.get("details", {}).get("database_hash_mismatches")}
        status = "completed"
    except Exception as error:  # recorded, never hidden
        result, status = {"error": f"{type(error).__name__}: {error}"}, "failed"
    finally:
        volume.commit()
    return {"command": f"{stage}:{arm}", "status": status, "gpu": GPU,
            "gpu_seconds": round(time.time() - started, 1), "result": result}


def _ledger(entries: list[dict]) -> float:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {
        "version": "sql-v12-modal-ledger-v1", "cost_cap_usd": COST_CAP_USD,
        "usd_per_gpu_second_estimate": USD_PER_GPU_SECOND, "jobs": []}
    for entry in entries:
        entry["cloud_gpu_cost_usd"] = round(entry.get("gpu_seconds", 0) * USD_PER_GPU_SECOND, 4)
        entry["recorded_at"] = datetime.now(timezone.utc).isoformat()
        ledger["jobs"].append(entry)
    ledger["total_cloud_gpu_cost_usd"] = round(sum(job["cloud_gpu_cost_usd"] for job in ledger["jobs"]), 4)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8", newline="\n")
    return ledger["total_cloud_gpu_cost_usd"]


def _upload_inputs():
    databases = set()
    for dataset in DATASETS:
        for line in (LOCAL / dataset).read_text(encoding="utf-8").splitlines():
            databases.add(Path(json.loads(line)["context"]["database"]))
    with volume.batch_upload(force=True) as batch:
        for dataset in DATASETS:
            batch.put_file(str(LOCAL / dataset), "/" + dataset)
            manifest = dataset.replace(".jsonl", ".manifest.json")
            batch.put_file(str(LOCAL / manifest), "/" + manifest)
        batch.put_file(str(LOCAL / "v2/sql-agent/bird/downloads/evaluation_ex.py"),
                       "/v2/sql-agent/bird/downloads/evaluation_ex.py")
        for database in sorted(databases):
            batch.put_file(str(database), f"/dbs/{database.stem}/{database.name}")
            descriptions = database.parent / "database_description"
            if descriptions.is_dir():
                batch.put_directory(str(descriptions), f"/dbs/{database.stem}/database_description")
    return len(databases)


def _pull():
    pulled = 0
    for entry in volume.listdir("v12", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = LOCAL / entry.path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        pulled += 1
    return pulled


@app.local_entrypoint()
def main(stage: str):
    spent = json.loads(LEDGER.read_text(encoding="utf-8"))["total_cloud_gpu_cost_usd"] if LEDGER.exists() else 0.0
    if stage not in {"pull", "setup"} and spent >= COST_CAP_USD:
        raise SystemExit(f"Cloud GPU cap of ${COST_CAP_USD} reached; stopping.")
    if stage == "setup":
        print("uploaded databases:", _upload_inputs())
        print(json.dumps(prepare.remote(), indent=2))
    elif stage in {"smoke", "development"}:
        arms = ["control", "full"] if stage == "smoke" else ["control", "rules", "notes", "full"]
        outcomes = [call.get() for call in [run_arm.spawn(stage, arm) for arm in arms]]
        print(json.dumps(outcomes, indent=2))
        print("spent", _ledger(outcomes))
    elif stage == "final":
        decision = LOCAL / "v12/sql-agent/development-decision.json"
        if not decision.is_file():
            raise SystemExit("Freeze the development decision locally before the locked final.")
        frozen = json.loads(decision.read_text(encoding="utf-8"))
        if not frozen.get("final_eligible"):
            raise SystemExit("Development selection did not permit the locked final.")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(decision), "/v12/sql-agent/development-decision.json")
        outcomes = [call.get() for call in [run_arm.spawn("final", arm) for arm in ("control", frozen["winner"])]]
        print(json.dumps(outcomes, indent=2))
        print("spent", _ledger(outcomes))
    elif stage == "pull":
        print("pulled", _pull(), "files")
    else:
        raise SystemExit("stage must be setup, smoke, development, final, or pull")
