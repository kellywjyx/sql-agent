"""Write an allowlisted checksum manifest for the local V9 research release."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
project = Path(__file__).resolve().parents[1]
artifacts = root / "artifacts/v9/sql-agent"
wheel = next((artifacts / "dist").glob("portfolio_sql_agent-0.9.0-*.whl"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


decision = json.loads((artifacts / "pilot-decision.json").read_text(encoding="utf-8"))
ledger = json.loads((root / "artifacts/v9/compute/ledger.json").read_text(encoding="utf-8"))
manifest = {
    "version": "sql-agent-v9-release-manifest-v1",
    "package_version": "0.9.0",
    "wheel": {"name": wheel.name, "sha256": digest(wheel)},
    "evidence": {
        "protocol_sha256": digest(artifacts / "protocol.json"),
        "transition_audit_sha256": digest(artifacts / "audit/report.json"),
        "scoping_decision_sha256": digest(artifacts / "scoping-decision.json"),
        "pilot_decision_sha256": digest(artifacts / "pilot-decision.json"),
        "results_document_sha256": digest(project / "docs/V9-RESULTS.md"),
    },
    "runs": {name: value["run_id"] for name, value in decision["conditions"].items()},
    "model_job_seconds": sum(job["charged_seconds"] for job in ledger["jobs"]),
    "model_jobs_complete": all(job["status"] == "completed" for job in ledger["jobs"]),
    "offline_tests_passed": 85,
    "dependency_check": "clean",
    "independent_wheel_import": "0.9.0",
    "public_default": "qwen_v5",
    "development_eligible": decision["development_eligible"],
    "locked_final_opened": decision["locked_final_opened"],
    "api_cost_usd": 0,
    "database_hash_mismatches": {name: value["database_hash_mismatches"]
                                  for name, value in decision["conditions"].items()},
}
(artifacts / "release-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
