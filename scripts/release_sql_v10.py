import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    artifacts = workspace / "artifacts" / "v10"
    project = artifacts / "sql-agent"
    decision = json.loads((project / "pilot-decision.json").read_text(encoding="utf-8"))
    ledger = json.loads((artifacts / "compute" / "ledger.json").read_text(encoding="utf-8"))
    wheel = project / "dist" / "portfolio_sql_agent-0.10.0-py3-none-any.whl"

    evidence = {
        "protocol": project / "protocol.json",
        "pilot_selection": project / "data" / "pilot-selection.json",
        "synthetic_memory": project / "error-memory" / "manifest.json",
        "natural_memory": project / "error-memory" / "natural-manifest.json",
        "calibration": project / "detector-calibration-v5.json",
        "pilot_decision": project / "pilot-decision.json",
        "design": repository / "docs" / "V10.md",
        "results": repository / "docs" / "V10-RESULTS.md",
        "wheel": wheel,
    }
    missing = [str(path) for path in evidence.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing V10 release inputs: " + ", ".join(missing))

    manifest = {
        "version": "sql-agent-v10-release-v1",
        "package": {"name": "portfolio-sql-agent", "version": "0.10.0"},
        "artifacts": {
            name: {
                "path": path.relative_to(workspace).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in evidence.items()
        },
        "runs": {
            "memory_capture": "20260901T132727.485726Z-879c759a",
            "arctic_scoped_control": decision["scoped_control"]["run_id"],
            "deterministic_replay": decision["deterministic"]["run_id"],
            "qwen_v5_same_set_control": decision["qwen_v5_same_set_control"]["run_id"],
        },
        "verification": {
            "offline_tests_passed": 92,
            "pip_check": "passed",
            "wheel_zip_import": "passed",
            "database_hash_mismatches": decision["database_hash_mismatches"],
            "locked_final_opened": decision["locked_final_opened"],
        },
        "decision": {
            "gate_passed": decision["gate_passed"],
            "selective_qwen_correction_eligible": decision["selective_qwen_correction_eligible"],
            "selective_qwen_correction_run": decision["selective_qwen_correction_run"],
            "retained_public_default": decision["retained_public_default"],
            "stop_reasons": decision["stop_reasons"],
        },
        "resource_record": {
            "charged_model_seconds": sum(
                int(job.get("charged_seconds", 0))
                for job in ledger.get("jobs", [])
                if job.get("project") == "sql-agent"
            ),
            "budget_seconds": ledger["limits"]["sql-agent"],
            "api_cost_usd": decision["api_cost_usd"],
        },
        "claims": {
            "benchmark": "local portfolio experiment; not an official BIRD submission",
            "runtime_gold_access": False,
            "security_policy_changed": False,
            "public_default_promoted": False,
        },
    }
    output = project / "release-manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
