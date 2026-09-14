"""V11.1: run the unchanged V11 QLoRA training functions on a Modal GPU.

Only the hardware changes (see docs/V11.1.md): training runs on an A100 and evaluation on an
A10G, both calling the unchanged sql_agent functions. Gate decisions are frozen locally.
Usage from the sql-agent directory:

    modal run scripts/modal_sql_v11.py --stage setup
    modal run scripts/modal_sql_v11.py --stage preflight
    modal run scripts/modal_sql_v11.py --stage pilot
    modal run scripts/modal_sql_v11.py --stage eval-pilot [--variants base,A,B]
    modal run scripts/modal_sql_v11.py --stage pull          # then freeze_pilot locally
    modal run scripts/modal_sql_v11.py --stage full --recipe A
    modal run scripts/modal_sql_v11.py --stage eval-development

The ledger applies the A100 rate to every job, so A10G costs are overstated (a conservative cap).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LOCAL_ARTIFACTS = PROJECT.parent / "artifacts"
WHEEL = LOCAL_ARTIFACTS / "releases/v0.4.0/wheels/portfolio_llm_evals-0.4.0-py3-none-any.whl"
REMOTE = Path("/artifacts")
MODEL_DIR = "v11/models/qwen2.5-coder-7b-instruct"
# V11.2 writes to its own tree; failed V11.1 attempts stay untouched under /artifacts/v11.
ROOT = REMOTE / "v11.2"
GPU = "A100-40GB"
# Estimated list price used only for the running cap; the Modal billing page is authoritative.
USD_PER_GPU_SECOND = 2.10 / 3600
COST_CAP_USD = 15.0
LEDGER = LOCAL_ARTIFACTS / "v11/compute/modal-ledger.json"

volume = modal.Volume.from_name("sql-v11-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.6", "trl==0.25.1", "peft==0.18.1", "bitsandbytes==0.49.2",
                 "accelerate==1.10.1", "datasets==4.4.1", "huggingface-hub==0.36.0",
                 "safetensors==0.8.0", "tokenizers==0.22.2", "sqlglot==30.0.1", "pydantic==2.12.5",
                 "numpy==2.5.2")
    .add_local_file(str(WHEEL), f"/wheels/{WHEEL.name}", copy=True)
    .run_commands(f"pip install /wheels/{WHEEL.name}")
    .env({"PYTHONPATH": "/src", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(PROJECT / "src/sql_agent"), "/src/sql_agent", ignore=["__pycache__"])
)
app = modal.App("sql-agent-v11-1", image=image)


@app.function(volumes={str(REMOTE): volume}, timeout=3600, cpu=4, memory=16384)
def download_model() -> dict:
    """Fetch the pinned revision and verify it against the recorded checksums."""
    from huggingface_hub import snapshot_download
    from sql_agent.v11_assets import verify
    from sql_agent.v11_data import MODEL_REPOSITORY, MODEL_REVISION
    target = REMOTE / MODEL_DIR
    snapshot_download(repo_id=MODEL_REPOSITORY, revision=MODEL_REVISION, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "LICENSE", "README.md"])
    manifest = verify(REMOTE)
    volume.commit()
    return {"revision": manifest["revision"], "verified_files": len(manifest["files"])}


@app.function(volumes={str(REMOTE): volume}, timeout=3600, cpu=4, memory=8192)
def prepare_tree() -> dict:
    """Copy V11.2 inputs into their own tree inside the volume; never overwrites or deletes."""
    import shutil
    from sql_agent.v11_assets import verify
    volume.reload()
    copied = {}
    for relative in ("v11/models/qwen2.5-coder-7b-instruct", "v11/sql-agent/data",
                     "v2/sql-agent/bird/downloads/evaluation_ex.py", "v11/sql-agent/eval/pilot/base"):
        source, target = REMOTE / relative, ROOT / relative
        if target.exists():
            copied[relative] = "exists"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        copied[relative] = "copied"
    manifest = verify(ROOT)
    volume.commit()
    return {"copied": copied, "verified_files": len(manifest["files"])}


def _use_recorded_asset_digests() -> None:
    """Reproduce the frozen dataset fingerprint without uploading 26 GB of training databases.

    Training reads only prompt messages. The per-database SHA-256 digests were computed locally
    from the real files, where every role's fingerprint was confirmed against its manifest.
    The frozen manifest hash must still match exactly; a missing digest fails closed.
    """
    from pathlib import PureWindowsPath
    import llm_evals.campaign as campaign
    import llm_evals.dataset as dataset
    recorded = json.loads((ROOT / "v11/compute/modal-asset-digests.json").read_text(encoding="utf-8"))["assets"]

    def fingerprint(cases):
        def portable(value):
            if isinstance(value, dict):
                return {key: portable(item) for key, item in value.items()}
            if isinstance(value, list):
                return [portable(item) for item in value]
            if isinstance(value, str) and PureWindowsPath(value).is_absolute():
                if value not in recorded:
                    raise ValueError(f"No locally verified digest for dataset asset: {value}")
                return {"asset_sha256": recorded[value]}
            return value
        return dataset.fingerprint([portable(case.model_dump()) for case in cases])

    campaign.dataset_fingerprint = fingerprint
    dataset.dataset_fingerprint = fingerprint


@app.function(volumes={str(REMOTE): volume}, gpu=GPU, timeout=3 * 3600, memory=32768)
def gpu_job(command: str, recipe: str = "A", attempt_id: str = "modal-a100-2") -> dict:
    from sql_agent import v11_training
    volume.reload()
    _use_recorded_asset_digests()
    started = time.time()
    try:
        if command == "preflight":
            result = v11_training.preflight(ROOT, attempt_id=attempt_id)
            adapter = ROOT / f"v11/sql-agent/preflight/{attempt_id}/adapter"
            result["reload"] = v11_training.reload_check(ROOT, adapter)
        elif command in {"pilot", "full"}:
            result = v11_training.train(ROOT, recipe, pilot=command == "pilot")
            final = ROOT / f"v11/sql-agent/training/{command}/{recipe}/final"
            result["reload"] = v11_training.reload_check(ROOT, final)
            result.pop("loss_history", None)
        else:
            raise ValueError(f"Unknown GPU command {command!r}")
        status = "completed"
    except Exception as error:  # recorded, never hidden
        result, status = {"error": f"{type(error).__name__}: {error}"}, "failed"
    finally:
        volume.commit()
    return {"command": command, "recipe": recipe, "status": status, "gpu": GPU,
            "gpu_seconds": round(time.time() - started, 1), "result": result}


def _link_databases(prepared: Path | None) -> None:
    """Expose uploaded databases under the frozen case paths without rewriting any dataset.

    Frozen cases store Windows paths; on Linux such a path is a single relative file name,
    so a symlink with that exact name in the working directory resolves to the uploaded copy.
    """
    import os
    work = Path("/work"); work.mkdir(exist_ok=True); os.chdir(work)
    sources = [ROOT / "v11/sql-agent/data/development.jsonl"] + ([prepared] if prepared else [])
    for source in sources:
        for line in source.read_text(encoding="utf-8").splitlines():
            frozen = json.loads(line)["context"]["database"]
            link = work / frozen
            if not link.exists():
                link.symlink_to(REMOTE / "dbs" / frozen.replace("\\", "/").split("/")[-1])


@app.function(volumes={str(REMOTE): volume}, gpu="A10G", timeout=4 * 3600, memory=32768, cpu=4)
def evaluate(stage: str, variant: str) -> dict:
    from sql_agent import v11_evaluation
    volume.reload()
    prepared = ROOT / "v11/sql-agent/data/eval-regression.jsonl"
    _link_databases(prepared if stage == "regression" else None)
    started = time.time()
    try:
        report = v11_evaluation.run(ROOT, stage, variant)
        result = {"run_id": report.get("run_id"), "metrics": report.get("metrics"),
                  "p95_seconds": report.get("p95_seconds"),
                  "database_hash_mismatches": report.get("details", {}).get("database_hash_mismatches")}
        status = "completed"
    except Exception as error:
        result, status = {"error": f"{type(error).__name__}: {error}"}, "failed"
    finally:
        volume.commit()
    return {"command": f"evaluate-{stage}", "recipe": variant, "status": status, "gpu": "A10G",
            "gpu_seconds": round(time.time() - started, 1), "result": result}


def _ledger(entries: list[dict]) -> float:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {
        "version": "sql-v11.1-modal-ledger-v1", "cost_cap_usd": COST_CAP_USD,
        "usd_per_gpu_second_estimate": USD_PER_GPU_SECOND, "jobs": []}
    for entry in entries:
        entry["cloud_gpu_cost_usd"] = round(entry.get("gpu_seconds", 0) * USD_PER_GPU_SECOND, 4)
        entry.setdefault("protocol", "V11.2")
        entry["recorded_at"] = datetime.now(timezone.utc).isoformat()
        ledger["jobs"].append(entry)
    ledger["total_cloud_gpu_cost_usd"] = round(sum(job["cloud_gpu_cost_usd"] for job in ledger["jobs"]), 4)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return ledger["total_cloud_gpu_cost_usd"]


def _spent() -> float:
    return json.loads(LEDGER.read_text(encoding="utf-8"))["total_cloud_gpu_cost_usd"] if LEDGER.exists() else 0.0


def _upload_inputs() -> None:
    data = LOCAL_ARTIFACTS / "v11/sql-agent/data"
    manifest = json.loads((LOCAL_ARTIFACTS / MODEL_DIR / "asset-manifest.json").read_text(encoding="utf-8"))
    # Hugging Face download-cache metadata is machine-specific; every model file is still verified.
    manifest["files"] = {name: digest for name, digest in manifest["files"].items() if not name.startswith(".cache/")}
    staged = LOCAL_ARTIFACTS / "v11/compute/modal-asset-manifest.json"
    staged.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with volume.batch_upload(force=True) as batch:
        for role in ("train", "validation", "development"):
            for suffix in (".jsonl", ".manifest.json"):
                batch.put_file(str(data / f"{role}{suffix}"), f"/v11/sql-agent/data/{role}{suffix}")
        batch.put_file(str(staged), f"/{MODEL_DIR}/asset-manifest.json")


def _pull(prefix: str) -> list[str]:
    pulled = []
    for entry in volume.listdir(prefix, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE or "/checkpoint-" in entry.path:
            continue
        target = LOCAL_ARTIFACTS / entry.path
        if target.exists():
            continue  # immutable local copies are never overwritten
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        pulled.append(entry.path)
    return pulled


@app.local_entrypoint()
def main(stage: str, recipe: str = "A", variants: str = ""):
    if stage != "pull" and _spent() >= COST_CAP_USD:
        raise SystemExit(f"Cloud GPU cap of ${COST_CAP_USD} reached; stopping.")
    if stage in {"preflight", "pilot", "full"}:
        digests = LOCAL_ARTIFACTS / "v11/compute/modal-asset-digests.json"
        roles = json.loads(digests.read_text(encoding="utf-8"))["roles"]
        if not all(role["matches"] for role in roles.values()):
            raise SystemExit("Local fingerprints do not match the frozen manifests; refusing to train.")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(digests), "/v11.2/v11/compute/modal-asset-digests.json")
    if stage == "setup":
        _upload_inputs()
        print(json.dumps(download_model.remote(), indent=2))
    elif stage == "setup-v11.2":
        print(json.dumps(prepare_tree.remote(), indent=2))
    elif stage == "preflight":
        outcome = gpu_job.remote("preflight")
        print(json.dumps(outcome, indent=2)); print("spent", _ledger([outcome]))
    elif stage == "pilot":
        calls = [gpu_job.spawn("pilot", name) for name in ("A", "B")]
        outcomes = [call.get() for call in calls]
        print(json.dumps(outcomes, indent=2)); print("spent", _ledger(outcomes))
    elif stage == "full":
        decision = LOCAL_ARTIFACTS / "v11.2/v11/sql-agent/pilot-decision.json"
        if not decision.is_file():
            raise SystemExit("Freeze the pilot decision locally before full training.")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(decision), "/v11.2/v11/sql-agent/pilot-decision.json")
        outcome = gpu_job.remote("full", recipe)
        print(json.dumps(outcome, indent=2)); print("spent", _ledger([outcome]))
    elif stage in {"eval-pilot", "eval-development", "eval-regression"}:
        name = stage.removeprefix("eval-")
        chosen = variants.split(",") if variants else {"pilot": ["base", "A", "B"],
                  "development": ["base", "selected"], "regression": ["selected"]}[name]
        if name != "pilot":
            decision = LOCAL_ARTIFACTS / "v11.2/v11/sql-agent/pilot-decision.json"
            with volume.batch_upload(force=True) as batch:
                batch.put_file(str(decision), "/v11.2/v11/sql-agent/pilot-decision.json")
        calls = [evaluate.spawn(name, variant) for variant in chosen]
        outcomes = [call.get() for call in calls]
        print(json.dumps(outcomes, indent=2)); print("spent", _ledger(outcomes))
    elif stage == "pull":
        pulled = (_pull("v11.2/v11/sql-agent/training") + _pull("v11.2/v11/sql-agent/preflight")
                  + _pull("v11.2/v11/sql-agent/eval"))
        exposure = LOCAL_ARTIFACTS / "v11.2/v11/compute/modal-exposure.jsonl"
        exposure.write_bytes(b"".join(volume.read_file("v11.2/v11/exposure.jsonl")))
        print(f"pulled {len(pulled)} files")
    else:
        raise SystemExit("stage must be setup, preflight, pilot, full, or pull")
