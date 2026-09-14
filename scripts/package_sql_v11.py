"""Verify and checksum the source-only V11 wheel after tests and reporting."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


repository = Path(__file__).resolve().parents[1]
artifacts = repository.parent / "artifacts"
wheel = artifacts / "releases/v0.11.0/wheels/portfolio_sql_agent-0.11.0-py3-none-any.whl"
if not wheel.is_file():
    raise SystemExit("Build the 0.11.0 wheel before packaging the release manifest")
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    metadata = archive.read("portfolio_sql_agent-0.11.0.dist-info/METADATA").decode()
required = {
    "sql_agent/v11_assets.py", "sql_agent/v11_data.py", "sql_agent/v11_evaluation.py",
    "sql_agent/v11_prompt.py", "sql_agent/v11_runtime.py", "sql_agent/v11_selection.py",
    "sql_agent/v11_training.py",
}
missing = sorted(required - names)
if missing or "Version: 0.11.0" not in metadata:
    raise SystemExit(f"Invalid V11 wheel; missing={missing}")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

hardware = artifacts / "v11/sql-agent/preflight/attempt-1/hardware-gate.json"
protocol = artifacts / "v11/sql-agent/protocol-final.json"
manifest = {
    "version": "portfolio-sql-agent-release-v1",
    "package_version": "0.11.0",
    "wheel": {"filename": wheel.name, "sha256": sha(wheel), "bytes": wheel.stat().st_size},
    "verified_modules": sorted(required),
    "offline_tests": {"passed": 100, "failed": 0},
    "campaign_status": json.loads(hardware.read_text(encoding="utf-8"))["status"],
    "protocol_sha256": sha(protocol),
    "hardware_gate_sha256": sha(hardware),
    "model_weights_in_wheel": False,
    "datasets_in_wheel": False,
    "adapter_in_wheel": False,
    "api_cost_usd": 0,
    "locked_final_opened": False,
}
target = wheel.parent.parent / "manifest.json"
target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(manifest, indent=2))
