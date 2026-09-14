"""Freeze V9 matched-stage and continuation decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_evals.campaign import immutable_json

from .v8_data import open_pilot
from .v8_evaluation import transitions
from .v8_selection import _interval, _run


def _condition(root: Path, cases, mode: str) -> dict:
    report, all_outcomes = _run(root, root / f"v9/sql-agent/eval/pilot/{mode}")
    outcomes = {case.id: all_outcomes[case.id] for case in cases}
    return {"run_id": report["run_id"], "capture_sha256": report["capture_sha256"],
            "execution_accuracy": sum(outcomes.values()) / len(outcomes),
            "completion": report["metrics"]["completion"], "p95_seconds": report["p95_seconds"],
            "database_hash_mismatches": report["details"]["database_hash_mismatches"],
            "evidence_utilization": report["details"]["evidence_utilization"],
            "outcomes": outcomes}


def freeze_scoping(root: Path) -> dict:
    root = root.resolve(); target = root / "v9/sql-agent/scoping-decision.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    cases = open_pilot(root); identifiers = [case.id for case in cases]
    v7 = json.loads((root / "v7/sql-agent/reproduction-decision.json").read_text(encoding="utf-8"))
    _, no_all = _run(root, root / "v7/sql-agent/eval/reproduction/reference_none",
                     v7["conditions"]["reference_none"]["run_id"])
    no = {identifier: no_all[identifier] for identifier in identifiers}
    conditions = {name: _condition(root, cases, name) for name in ["p0_cache_v2", "scoped"]}
    for value in conditions.values():
        value["transitions_vs_no_evidence"] = transitions(no, value["outcomes"])
        value["comparison_vs_no_evidence"] = _interval(cases, no, value["outcomes"], seed=20260913)
        value.pop("outcomes")
    p0, scoped = conditions["p0_cache_v2"], conditions["scoped"]
    p0_t, scoped_t = p0["transitions_vs_no_evidence"], scoped["transitions_vs_no_evidence"]
    probes_eligible = bool(scoped["execution_accuracy"] >= p0["execution_accuracy"] and
                           scoped["completion"] >= p0["completion"] and
                           scoped_t["correct_to_wrong"] < p0_t["correct_to_wrong"] and
                           not scoped["database_hash_mismatches"])
    result = {"version": "sql-v9-scoping-decision-v1", "case_count": len(cases),
              "no_evidence_ex": sum(no.values()) / len(no), "conditions": conditions,
              "probes_eligible": probes_eligible,
              "probe_rule": "scoped EX/completion no worse than cache-v2 and fewer regressions versus no evidence",
              "locked_final_opened": False, "api_cost_usd": 0}
    immutable_json(target, result); return result


def freeze_pilot(root: Path) -> dict:
    root = root.resolve(); target = root / "v9/sql-agent/pilot-decision.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    scoping = freeze_scoping(root)
    cases = open_pilot(root); identifiers = [case.id for case in cases]
    v7 = json.loads((root / "v7/sql-agent/reproduction-decision.json").read_text(encoding="utf-8"))
    _, no_all = _run(root, root / "v7/sql-agent/eval/reproduction/reference_none",
                     v7["conditions"]["reference_none"]["run_id"])
    no = {identifier: no_all[identifier] for identifier in identifiers}
    conditions = {name: value for name, value in scoping["conditions"].items()}
    if scoping["probes_eligible"]:
        probed = _condition(root, cases, "probed")
        probed["transitions_vs_no_evidence"] = transitions(no, probed["outcomes"])
        probed["comparison_vs_no_evidence"] = _interval(cases, no, probed["outcomes"], seed=20260913)
        probed.pop("outcomes"); conditions["probed"] = probed
    protocol = json.loads((root / "v9/sql-agent/protocol.json").read_text(encoding="utf-8"))
    eligible = []
    for name, value in conditions.items():
        transition = value["transitions_vs_no_evidence"]
        gate = protocol["pilot_gate"]
        net = transition["wrong_to_correct"] - transition["correct_to_wrong"]
        value["gate_passed"] = bool(
            transition["wrong_to_correct"] >= gate["recoveries"] and
            transition["correct_to_wrong"] <= gate["regressions_max"] and net >= gate["net_recoveries"] and
            value["execution_accuracy"] - sum(no.values()) / len(no) >= gate["ex_lift"] and
            value["completion"] >= gate["completion"] and value["p95_seconds"] < gate["p95_seconds_max"] and
            not value["database_hash_mismatches"])
        if value["gate_passed"]:
            eligible.append((value["execution_accuracy"], name))
    selected = max(eligible)[1] if eligible else None
    result = {"version": "sql-v9-pilot-decision-v1", "case_count": len(cases),
              "no_evidence_ex": sum(no.values()) / len(no), "conditions": conditions,
              "selected": selected, "development_eligible": selected is not None,
              "stop_reasons": [] if selected else ["No condition passed the frozen recovery/regression/latency gate"],
              "retained_public_default": "qwen_v5", "locked_final_opened": False, "api_cost_usd": 0}
    immutable_json(target, result); return result
