"""Freeze the V10 pilot and promotion decision."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals.campaign import immutable_json

from .v8_selection import _interval, _run
from .v10_data import open_pilot


def freeze(root: Path) -> dict:
    root = root.resolve(); target = root / "v10/sql-agent/pilot-decision.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    cases = open_pilot(root); ids = [case.id for case in cases]
    control_report, control_all = _run(root, root / "v10/sql-agent/eval/pilot/scoped_control")
    repaired_report, repaired_all = _run(root, root / "v10/sql-agent/eval/pilot/deterministic")
    qwen_report, qwen_all = _run(root, root / "v10/sql-agent/eval/pilot/qwen_v5_control")
    control, repaired, qwen = ({identifier: values[identifier] for identifier in ids}
                               for values in [control_all, repaired_all, qwen_all])
    score = lambda rows: sum(rows.values()) / len(rows)
    control_ex, repaired_ex, qwen_ex = score(control), score(repaired), score(qwen)
    details = repaired_report["details"]
    protocol = json.loads((root / "v10/sql-agent/protocol.json").read_text(encoding="utf-8"))
    gate = protocol["pilot_gate"]
    detection_precision = details["detection_precision"]
    repair_success = details["repair_success"]
    reasons = []
    if repaired_ex - control_ex < gate["ex_lift"]: reasons.append("EX lift is below +0.075")
    if details["net_corrections"] < gate["net_corrections"]: reasons.append("net corrections are below 3")
    if details["transitions"]["correct_to_wrong"] > gate["correct_query_regressions_max"]:
        reasons.append("more than one correct query regressed")
    if detection_precision is None or detection_precision < gate["detection_precision"]:
        reasons.append("fresh-pilot detection precision is unscorable or below 0.80")
    if repair_success is None or repair_success < gate["repair_success"]:
        reasons.append("fresh-pilot repair success is unscorable or below 0.50")
    if repaired_report["metrics"]["completion"] < gate["completion"]: reasons.append("completion below 0.95")
    if details["end_to_end_p95_seconds"] >= gate["p95_seconds_max"]: reasons.append("p95 reached 20 seconds")
    if details["database_hash_mismatches"]: reasons.append("database hash mismatch")
    qwen_activation = bool(repaired_ex - control_ex < gate["ex_lift"] and
                           detection_precision is not None and detection_precision >= .80 and
                           details["attempted_corrections"] > 0 and details["wrong_attempts"] > details["transitions"]["wrong_to_correct"])
    result = {"version": "sql-v10-pilot-decision-v1", "case_count": len(cases),
              "scoped_control": {"run_id": control_report["run_id"], "execution_accuracy": control_ex,
                                  "completion": control_report["metrics"]["completion"],
                                  "p95_seconds": control_report["p95_seconds"]},
              "deterministic": {"run_id": repaired_report["run_id"], "execution_accuracy": repaired_ex,
                                "ex_lift": repaired_ex - control_ex,
                                "completion": repaired_report["metrics"]["completion"],
                                "end_to_end_p95_seconds": details["end_to_end_p95_seconds"],
                                "transitions": details["transitions"],
                                "attempted_corrections": details["attempted_corrections"],
                                "detection_precision": detection_precision, "repair_success": repair_success,
                                "comparison": _interval(cases, control, repaired, seed=20260917)},
              "qwen_v5_same_set_control": {"run_id": qwen_report["run_id"], "execution_accuracy": qwen_ex,
                                           "completion": qwen_report["metrics"]["completion"],
                                           "p95_seconds": qwen_report["p95_seconds"],
                                           "comparison_vs_arctic": _interval(cases, control, qwen, seed=20260917)},
              "gate_passed": not reasons, "stop_reasons": reasons,
              "selective_qwen_correction_eligible": qwen_activation,
              "selective_qwen_correction_run": False,
              "retained_public_default": "qwen_v5", "locked_final_opened": False,
              "database_hash_mismatches": {"scoped": control_report["details"]["database_hash_mismatches"],
                                             "deterministic": details["database_hash_mismatches"],
                                             "qwen": qwen_report["details"]["database_hash_mismatches"]},
              "api_cost_usd": 0,
              "conclusion": "The high-precision detector transferred too narrowly to activate correction on the fresh pilot."}
    immutable_json(target, result); return result
