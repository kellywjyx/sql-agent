"""Approval-gated local BIRD-Talon critic support."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_evals.local import OllamaClient

from .generation import _extract_sql
from .guardrails import validate
from .v7_critic import structural_signature
from .v7_generation import talon_repair_messages


UPSTREAM = "birdsql/BIRD-Talon-7b"
ALLOWED_COMPONENT_CHANGES = {
    "projection_count": {"projection_count", "projections", "columns"},
    "projection_bindings": {"projections", "columns"},
    "filter_bindings": {"predicates", "columns"},
    "literal_values": {"predicates"},
    "aggregate_functions": {"aggregates", "projections"},
    "grouping_grain": {"grouping"}, "global_scalar_grain": {"grouping"},
    "having": {"having", "predicates"}, "distinct": {"distinct"},
    "ordering": {"ordering"}, "ordering_direction": {"ordering"},
    "limit": {"limit"}, "unnecessary_limit": {"limit"},
    "subquery": {"subqueries"}, "set_operation": {"set_operation"},
    "required_tables": {"tables", "joins", "columns"},
}


def load_asset(artifacts: Path) -> tuple[dict, OllamaClient]:
    path = artifacts.resolve() / "sql-agent/models/talon/asset-manifest.json"
    if not path.exists():
        raise RuntimeError("BIRD-Talon is gated and not installed. Record approval, license, revision, GGUF SHA-256, and Ollama digest before use.")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"upstream_repository", "upstream_revision", "license", "gguf_sha256", "ollama_name", "ollama_digest",
                "download_approved_at"}
    if not required <= value.keys() or value["upstream_repository"] != UPSTREAM:
        raise RuntimeError("Talon asset manifest is incomplete or points at an unexpected upstream repository")
    if len(value["gguf_sha256"]) != 64 or not value["license"]:
        raise RuntimeError("Talon license or GGUF checksum is not auditable")
    client = OllamaClient(model=value["ollama_name"])
    ready = client.check()
    if ready.get("digest") != value["ollama_digest"]:
        raise RuntimeError("Installed Talon Ollama digest differs from the approved asset manifest")
    return value, client


def gate(artifacts: Path) -> dict:
    selection_path = artifacts.resolve() / "v7/sql-agent/selection.json"
    if not selection_path.exists():
        return {"eligible": False, "reasons": ["V7 development selection is not frozen"]}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metrics = selection["metrics"]
    gap = metrics["candidate_oracle_ex"] - metrics["selected_ex"]
    reasons = []
    if metrics["candidate_oracle_ex"] < .65:
        reasons.append("candidate oracle EX below 0.65")
    if gap < .10 and metrics.get("deterministic_repair_gain", 0) >= .03:
        reasons.append("selection gap and deterministic repair do not activate Talon")
    try:
        asset, _ = load_asset(artifacts)
    except RuntimeError as error:
        reasons.append(str(error))
        asset = None
    return {"eligible": not reasons, "reasons": reasons, "selection_gap": gap,
            "asset": asset, "contamination_note": "BIRD ecosystem critic; not an independent clean benchmark model"}


def repair(client, question: str, schema: str, evidence_packet: str, grounded, candidate_sql: str,
           critic: dict, result_summary: dict) -> dict:
    diagnostics = critic.get("actionable_diagnostics") or []
    if len(diagnostics) != 1 or diagnostics[0] not in ALLOWED_COMPONENT_CHANGES:
        return {"accepted": False, "reason": "Talon requires one allowlisted diagnosed component"}
    completion = client.chat(talon_repair_messages(question, schema, evidence_packet, grounded, candidate_sql,
                                                    critic, result_summary), max_tokens=700, context_tokens=8192)
    revised = validate(_extract_sql(completion.text))
    before, after = structural_signature(candidate_sql), structural_signature(revised)
    changed = {name for name in before if before[name] != after[name]}
    allowed = ALLOWED_COMPONENT_CHANGES[diagnostics[0]]
    if not changed or not changed <= allowed:
        return {"accepted": False, "reason": "broad or empty Talon rewrite rejected",
                "component": diagnostics[0], "changed_dimensions": sorted(changed),
                "raw_response_sha256": hashlib.sha256(completion.text.encode()).hexdigest()}
    return {"accepted": True, "component": diagnostics[0], "sql": revised,
            "changed_dimensions": sorted(changed), "raw_response": completion.text,
            "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens,
            "generation_seconds": completion.generation_seconds}
