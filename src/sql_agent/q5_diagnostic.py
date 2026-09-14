"""Paired Q4/Q5 quantization diagnostic on the approved exposed 20 cases."""
from __future__ import annotations

import json
import math
from pathlib import Path

from llm_evals import Metric
from llm_evals.campaign import grouped_comparison, immutable_json

from .v7_data import open_cases
from .v7_evaluation import METRIC_VERSION, V7Scoring


def _run(root: Path, mode: str, run_id: str | None = None):
    folder = root / f"v7/sql-agent/eval/reproduction/{mode}"
    if run_id is None:
        run_id = json.loads((folder / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run = folder / run_id
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if report.get("capture_status") != "complete":
        raise ValueError(f"Incomplete Q5 diagnostic input: {mode}/{run_id}")
    return report, predictions


def _p95(predictions: list[dict]) -> float:
    values = sorted(float(row.get("latency_seconds", 0)) for row in predictions)
    return values[max(0, math.ceil(.95 * len(values)) - 1)] if values else 0


def report(root: Path) -> dict:
    root = root.resolve()
    target = root / "v7/sql-agent/q5-diagnostic.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    reproduction = json.loads((root / "v7/sql-agent/reproduction-decision.json").read_text(encoding="utf-8"))
    if not reproduction.get("q5_matched_pilot_eligible"):
        raise ValueError("The frozen reproduction gate did not authorize a Q5 pilot")
    q4_run_id = reproduction["conditions"]["reference_oracle"]["run_id"]
    q4_report, q4_all = _run(root, "reference_oracle", q4_run_id)
    q5_report, q5_predictions = _run(root, "reference_q5_oracle")
    cases = open_cases(root, "reproduction")[:20]
    q4_by_id = {row["id"]: row for row in q4_all}
    q5_by_id = {row["id"]: row for row in q5_predictions}
    ids = [case.id for case in cases]
    if set(ids) - q4_by_id.keys() or set(ids) != set(q5_by_id):
        raise ValueError("Q4 and Q5 captures are not aligned to the approved 20 cases")
    q4 = [q4_by_id[case.id] for case in cases]
    q5 = [q5_by_id[case.id] for case in cases]
    scorer = V7Scoring(cases, official_scorer_path=root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    q4_correct = [scorer.correct(case, prediction) for case, prediction in zip(cases, q4)]
    q5_correct = [scorer.correct(case, prediction) for case, prediction in zip(cases, q5)]
    transitions = {
        "q4_correct_q5_correct": sum(left and right for left, right in zip(q4_correct, q5_correct)),
        "q4_wrong_q5_correct": sum(not left and right for left, right in zip(q4_correct, q5_correct)),
        "q4_correct_q5_wrong": sum(left and not right for left, right in zip(q4_correct, q5_correct)),
        "q4_wrong_q5_wrong": sum(not left and not right for left, right in zip(q4_correct, q5_correct)),
    }
    metric = Metric("execution_accuracy_v7", scorer.execution_accuracy, version=METRIC_VERSION)
    interval = grouped_comparison(metric, cases, q4, q5, group_key="bootstrap_group",
                                  samples=2000, seed=20260907)
    q4_ex, q5_ex = sum(q4_correct) / 20, sum(q5_correct) / 20
    regressions, recoveries = transitions["q4_correct_q5_wrong"], transitions["q4_wrong_q5_correct"]
    delta = q5_ex - q4_ex
    retain = delta >= .03 or (recoveries >= 2 and regressions == 0)
    interpretation = ("substantial paired improvement; Q4 suppressed capability" if delta >= .10 else
                      "small quantization effect; insufficient to explain the published gap" if retain else
                      "Q5 roughly unchanged; quantization is not the primary bottleneck")
    result = {
        "version": "sql-v7-q5-diagnostic-v1",
        "scope": "20 exposed reproduction-audit cases; reference-like prompt with oracle evidence",
        "changed_variable": "Arctic GGUF Q4_K_M to Q5_K_M",
        "q4": {"run_id": q4_report["run_id"], "execution_accuracy": q4_ex,
               "completion": sum(row.get("status") == "completed" for row in q4) / 20,
               "p95_seconds": _p95(q4), "model": q4_report["identity"]["model"],
               "digest": q4_report["identity"]["digest"]},
        "q5": {"run_id": q5_report["run_id"], "execution_accuracy": q5_ex,
               "completion": q5_report["metrics"]["completion"], "p95_seconds": q5_report["p95_seconds"],
               "model": q5_report["identity"]["model"], "digest": q5_report["identity"]["digest"]},
        "transitions": transitions,
        "paired_interval": interval,
        "retain_q5_for_future_diagnostics": retain,
        "interpretation": interpretation,
        "database_hash_mismatches": q5_report.get("details", {}).get("database_hash_mismatches", []),
        "official_gold_self_consistency": q5_report["metrics"]["official_gold_self_consistency"],
        "api_charge_usd": 0,
        "restart_v7_campaign": False,
        "run_talon": False,
        "locked_final_opened": False,
    }
    immutable_json(target, result)
    return result


def case_transitions(root: Path) -> dict:
    """Write answer-aware transition rows without copying gold SQL into the artifact."""
    import hashlib

    root = root.resolve()
    target = root / "v7/sql-agent/q5-case-transitions.jsonl"
    manifest_path = root / "v7/sql-agent/q5-case-transitions.manifest.json"
    if target.exists() or manifest_path.exists():
        if not target.exists() or not manifest_path.exists():
            raise ValueError("Incomplete immutable Q5 transition artifact")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    reproduction = json.loads((root / "v7/sql-agent/reproduction-decision.json").read_text(encoding="utf-8"))
    q4_run_id = reproduction["conditions"]["reference_oracle"]["run_id"]
    _, q4_all = _run(root, "reference_oracle", q4_run_id)
    _, q5_all = _run(root, "reference_q5_oracle")
    cases = open_cases(root, "reproduction")[:20]
    q4_by_id, q5_by_id = ({row["id"]: row for row in rows} for rows in (q4_all, q5_all))
    if any(case.id not in q4_by_id or case.id not in q5_by_id for case in cases):
        raise ValueError("Q4 and Q5 case transition captures are not aligned")
    scorer = V7Scoring(cases, official_scorer_path=root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    rows = []
    for case in cases:
        q4, q5 = q4_by_id[case.id], q5_by_id[case.id]
        left, right = scorer.correct(case, q4), scorer.correct(case, q5)
        transition = ("q4_correct_q5_correct" if left and right else
                      "q4_wrong_q5_correct" if right else
                      "q4_correct_q5_wrong" if left else "q4_wrong_q5_wrong")
        rows.append({"id": case.id, "db_id": case.metadata["db_id"],
                     "difficulty": case.metadata.get("difficulty"), "transition": transition,
                     "q4": {"correct": left, "status": q4.get("status"), "sql": q4.get("sql")},
                     "q5": {"correct": right, "status": q5.get("status"), "sql": q5.get("sql")}})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"version": "sql-v7-q5-case-transitions-v1", "count": len(rows),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "contains_gold_sql": False, "locked_final_opened": False}
    immutable_json(manifest_path, manifest)
    return manifest
