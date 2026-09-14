"""Freeze the matched V8 pilot decision from capture-time evaluator outcomes."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from llm_evals.campaign import immutable_json

from .v8_evaluation import oracle_gap_recovery, transitions


def _run(root: Path, folder: Path, run_id: str | None = None) -> tuple[dict, dict[str, bool]]:
    if run_id is None:
        run_id = json.loads((folder / "latest.json").read_text(encoding="utf-8"))["run_id"]
    path = folder / run_id
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if report.get("capture_status") != "complete":
        raise ValueError(f"Incomplete V8 input: {folder.name}/{run_id}")
    outcomes = {row["id"]: row["category"] == "correct" for row in
                (json.loads(line) for line in (path / "failure-analysis.jsonl").read_text(
                    encoding="utf-8").splitlines())}
    report["capture_sha256"] = hashlib.sha256((path / "predictions.jsonl").read_bytes()).hexdigest()
    return report, outcomes


def _interval(cases, left, right, *, samples=2000, seed=20260911):
    groups = {}
    for case in cases:
        groups.setdefault(case.metadata["db_id"], []).append(case.id)
    rng, values, names = random.Random(seed), [], sorted(groups)
    for _ in range(samples):
        selected = [identifier for name in rng.choices(names, k=len(names)) for identifier in groups[name]]
        values.append(sum(right[i] - left[i] for i in selected) / len(selected))
    values.sort()
    delta = sum(right[i] - left[i] for i in left) / len(left)
    return {"delta": delta, "low": values[int(.025 * samples)], "high": values[int(.975 * samples)],
            "samples": samples, "groups": len(groups), "method": "paired database-group bootstrap"}


def freeze(root: Path) -> dict:
    from .v8_data import open_pilot
    root = root.resolve()
    target = root / "v8/sql-agent/pilot-decision.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    cases = open_pilot(root)
    ids = [case.id for case in cases]
    protocol = json.loads((root / "v8/sql-agent/protocol.json").read_text(encoding="utf-8"))
    controls = {}
    for name, mode in [("no_evidence", "reference_none"), ("oracle", "reference_oracle")]:
        report, outcomes = _run(root, root / f"v7/sql-agent/eval/reproduction/{mode}",
                                protocol["controls"][f"{name}_run"] if f"{name}_run" in protocol["controls"] else
                                protocol["controls"]["oracle_evidence_run"])
        controls[name] = {"report": report, "outcomes": {i: outcomes[i] for i in ids}}
    conditions = {}
    for mode in protocol["ablation_conditions"]:
        report, outcomes = _run(root, root / f"v8/sql-agent/eval/pilot/{mode}")
        aligned = {i: outcomes[i] for i in ids}
        ex = sum(aligned.values()) / len(ids)
        transition = transitions(controls["no_evidence"]["outcomes"], aligned)
        conditions[mode] = {"run_id": report["run_id"], "capture_sha256": report["capture_sha256"],
                            "execution_accuracy": ex, "completion": report["metrics"]["completion"],
                            "p95_seconds": report["p95_seconds"], "transitions": transition,
                            "comparison": _interval(cases, controls["no_evidence"]["outcomes"], aligned),
                            "database_hash_mismatches": report["details"]["database_hash_mismatches"]}
    no_ex = sum(controls["no_evidence"]["outcomes"].values()) / len(ids)
    oracle_ex = sum(controls["oracle"]["outcomes"].values()) / len(ids)
    for value in conditions.values():
        value["oracle_gap_recovery"] = oracle_gap_recovery(no_ex, oracle_ex, value["execution_accuracy"])
        value["gate_c_passed"] = bool(
            value["execution_accuracy"] - no_ex >= protocol["gates"]["pilot_min_ex_lift"] and
            value["transitions"]["wrong_to_correct"] > value["transitions"]["correct_to_wrong"] and
            not value["database_hash_mismatches"])
    eligible = [(value["execution_accuracy"], mode) for mode, value in conditions.items()
                if value["gate_c_passed"]]
    selected = max(eligible)[1] if eligible else None
    result = {"version": "sql-v8-pilot-decision-v1", "case_count": len(ids),
              "controls": {"no_evidence_ex": no_ex, "oracle_evidence_ex": oracle_ex},
              "conditions": conditions, "selected": selected, "gate_c_passed": selected is not None,
              "selection_rule": "highest capture-time EX among Gate-C-eligible evidence modes",
              "evaluator_note": "Paired outcomes use each immutable run's checksum-pinned capture-time evaluator; expected SQL never entered model inputs.",
              "locked_final_opened": False, "api_cost_usd": 0}
    immutable_json(target, result)
    return result


def freeze_development(root: Path) -> dict:
    from .v8_data import open_cases
    root = root.resolve()
    target = root / "v8/sql-agent/development-decision.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    cases = open_cases(root)
    ids = [case.id for case in cases]
    labels = {row["id"]: row["group"] for row in
              (json.loads(line) for line in (root / "v8/sql-agent/decomposition.jsonl").read_text(
                  encoding="utf-8").splitlines())}
    protocol = json.loads((root / "v8/sql-agent/protocol.json").read_text(encoding="utf-8"))
    no_report, no_all = _run(root, root / "v7/sql-agent/eval/reproduction/reference_none",
                             protocol["controls"]["no_evidence_run"])
    oracle_report, oracle_all = _run(root, root / "v7/sql-agent/eval/reproduction/reference_oracle",
                                     protocol["controls"]["oracle_evidence_run"])
    candidate_report, candidate_all = _run(root, root / "v8/sql-agent/eval/development/generated_all")
    no, oracle = {i: no_all[i] for i in ids}, {i: oracle_all[i] for i in ids}
    candidate = {i: candidate_all[i] for i in ids}
    score = lambda rows: sum(rows.values()) / len(rows)
    no_ex, oracle_ex, candidate_ex = score(no), score(oracle), score(candidate)
    transition = transitions(no, candidate)
    groups = {}
    for group in "ABCD":
        selected = [identifier for identifier in ids if labels[identifier] == group]
        if selected:
            groups[group] = {"cases": len(selected),
                             "no_evidence_ex": sum(no[i] for i in selected) / len(selected),
                             "generated_ex": sum(candidate[i] for i in selected) / len(selected),
                             "oracle_ex": sum(oracle[i] for i in selected) / len(selected),
                             "recoveries": [i for i in selected if not no[i] and candidate[i]],
                             "regressions": [i for i in selected if no[i] and not candidate[i]]}
    delta = candidate_ex - no_ex
    reasons = []
    if delta < protocol["gates"]["pilot_min_ex_lift"]:
        reasons.append("EX lift is below the frozen +0.075 continuation target")
    if transition["wrong_to_correct"] <= transition["correct_to_wrong"]:
        reasons.append("recoveries do not exceed regressions")
    if candidate_report["metrics"]["completion"] < .95:
        reasons.append("completion is below 0.95")
    if candidate_report["p95_seconds"] > 20:
        reasons.append("p95 latency exceeds 20 seconds")
    if candidate_report["details"]["database_hash_mismatches"]:
        reasons.append("database hash mismatch")
    result = {"version": "sql-v8-development-decision-v1", "case_count": len(ids),
              "run_id": candidate_report["run_id"], "capture_sha256": candidate_report["capture_sha256"],
              "no_evidence_ex": no_ex, "generated_evidence_ex": candidate_ex,
              "oracle_evidence_ex": oracle_ex, "ex_lift": delta,
              "oracle_gap_recovery": oracle_gap_recovery(no_ex, oracle_ex, candidate_ex),
              "completion": candidate_report["metrics"]["completion"],
              "p95_seconds": candidate_report["p95_seconds"], "transitions": transition,
              "comparison": _interval(cases, no, candidate), "groups": groups,
              "evidence_metrics": candidate_report["details"]["evidence_metrics"],
              "database_hash_mismatches": candidate_report["details"]["database_hash_mismatches"],
              "gate_d_passed": not reasons, "stopped": bool(reasons), "stop_reasons": reasons,
              "retained_public_default": "qwen_v5", "experimental_specialist": "arctic_q4",
              "locked_final_opened": False, "api_cost_usd": 0,
              "conclusion": "Generated evidence helps the evidence-limited group but is not yet precise enough to avoid regressions on evidence-unnecessary cases."}
    immutable_json(target, result)
    return result
