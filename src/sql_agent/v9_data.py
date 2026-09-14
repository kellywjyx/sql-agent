"""Freeze the V9 campaign to the already exposed V8 cases."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals.campaign import immutable_json


PROTOCOL_VERSION = "sql-v9-protocol-v1"


def prepare(root: Path) -> dict:
    root = root.resolve()
    target = root / "v9/sql-agent"
    path = target / "protocol.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    v8 = json.loads((root / "v8/sql-agent/development-decision.json").read_text(encoding="utf-8"))
    protocol = {
        "version": PROTOCOL_VERSION,
        "source": "immutable V8 20-case pilot and 40-case exposed development only",
        "v8_execution_accuracy": v8["generated_evidence_ex"],
        "public_default": "qwen_v5",
        "generator": "Arctic Q4 reference-like single path",
        "cache_control": "sql-v8-knowledge-cache-v2",
        "api_cost_budget_usd": 0,
        "model_job_budget_seconds": 3600,
        "max_probes": 4,
        "probe_timeout_seconds": 1,
        "locked_final_opened": False,
        "forbidden": ["locked final", "multi-candidate generation", "Talon", "Q5", "external API", "gold SQL in runtime"],
        "stages": ["p0_cache_v2", "b_scoped_evidence", "c_verified_probes", "d_verified_ir"],
        "pilot_gate": {"recoveries": 6, "regressions_max": 1, "net_recoveries": 5,
                       "ex_lift": .10, "completion": .95, "p95_seconds_max": 20,
                       "database_mutations": 0},
        "development_gate": {"recoveries": 6, "regressions_max": 1, "net_recoveries": 5,
                             "ex_lift": .10, "completion": .95, "p95_seconds_max": 20,
                             "database_mutations": 0},
        "stage_rules": {
            "probes": "only if scoped evidence reduces regressions without reducing EX or completion",
            "verified_ir": "only if probes improve EX with no new regression",
            "development": "only the pilot-selected configuration after its frozen gate passes",
        },
    }
    immutable_json(path, protocol)
    return protocol
