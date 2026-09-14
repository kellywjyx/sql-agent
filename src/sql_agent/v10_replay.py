"""Replay the frozen V10 control with selective deterministic correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

from llm_evals import run_suite

from .execution import execute_sql
from .guardrails import validate
from .integrity import changed_databases, snapshot_database_hashes
from .v7_evaluation import V7Scoring, structural_error_families
from .v8_selection import _run
from .v10_calibration import actual_families
from .v10_correction import detect, localized_patch, retrieve
from .v10_data import open_pilot
from .v10_signature import build_signature


REPLAY_VERSION = "sql-v10-deterministic-replay-v1"


def main():
    root = Path(__file__).resolve().parents[3] / "artifacts"
    cases = open_pilot(root)
    control_report, _ = _run(root, root / "v10/sql-agent/eval/pilot/scoped_control")
    control_dir = root / "v10/sql-agent/eval/pilot/scoped_control" / control_report["run_id"]
    controls = {row["id"]: row for row in
                (json.loads(line) for line in (control_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    calibration = json.loads((root / "v10/sql-agent/detector-calibration-v5.json").read_text(encoding="utf-8"))
    enabled = set(calibration["enabled_families"])
    if not enabled:
        raise SystemExit("No detector family passed calibration; deterministic replay is blocked")
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    before = snapshot_database_hashes(cases)
    memory_path = root / "v10/sql-agent/error-memory/combined.jsonl"

    def predict(question, context):
        original = controls[context["case_id"]]
        natural = question.split("Benchmark evidence:", 1)[0].strip()
        detection = detect(natural, original.get("sql", ""), original.get("evidence_packet", []), original)
        base = {**original, "original_sql": original.get("sql"), "original_status": original.get("status"),
                "original_latency_seconds": original.get("latency_seconds", original.get("generation_seconds", 0)),
                "detector": detection.model_dump(), "correction_examples": [], "correction_events": [],
                "pipeline_profile": "v10_deterministic_correction"}
        if detection.abstained or detection.error_type not in enabled:
            return {**base, "selection_reason": "detector_abstained", "correction_attempted": False}
        signature = build_signature(natural, original.get("evidence_packet", []), original["sql"], original)
        examples = retrieve(memory_path, signature, detection.error_type)
        patch = localized_patch(original["sql"], detection)
        if not patch:
            return {**base, "correction_examples": examples, "selection_reason": "localized_patch_unavailable",
                    "correction_attempted": False}
        started = perf_counter()
        try:
            sql = validate(patch["sql"])
            result = execute_sql(Path(context["database"]), sql, timeout=10)
            elapsed = perf_counter() - started
            event = {**patch, "error_type": detection.error_type, "confidence": detection.confidence,
                     "accepted": True, "execution_status": "completed", "seconds": elapsed}
            original_candidate = {"path": "arctic_scoped", "sql": original.get("sql"),
                                  "status": "executed" if original.get("status") == "completed" else "failed",
                                  "result": {"columns": original.get("columns", []), "rows": original.get("rows", [])}}
            patched_candidate = {"path": "deterministic_patch", "sql": sql, "status": "executed", "result": result}
            return {**base, **result, "sql": sql, "output": result["rows"], "status": "completed",
                    "candidates": [original_candidate, patched_candidate], "correction_examples": examples,
                    "correction_events": [event], "correction_attempted": True,
                    "correction_seconds": elapsed, "selection_reason": "calibrated_local_patch",
                    "needs_review": False}
        except (ValueError, RuntimeError, TimeoutError) as error:
            elapsed = perf_counter() - started
            event = {**patch, "error_type": detection.error_type, "confidence": detection.confidence,
                     "accepted": False, "execution_status": "failed", "seconds": elapsed,
                     "error": str(error)[:1000]}
            return {**base, "correction_examples": examples, "correction_events": [event],
                    "correction_attempted": True, "correction_seconds": elapsed,
                    "selection_reason": "patch_execution_failed_original_retained", "needs_review": True}

    run_cases = [case.model_copy(update={"context": {**case.context, "case_id": case.id}}) for case in cases]

    def details(used_cases, predictions):
        transitions = {"wrong_to_correct": 0, "correct_to_wrong": 0,
                       "wrong_to_wrong": 0, "correct_to_correct": 0}
        detected = hits = wrong_attempts = repair_successes = 0
        for case, prediction in zip(cases, predictions):
            original = controls[case.id]
            before_correct, after_correct = scorer.correct(case, original), scorer.correct(case, prediction)
            attempted = prediction.get("correction_attempted", False)
            if attempted:
                detected += 1
                actual = actual_families(case.expected["sql"], original.get("sql"))
                hits += prediction.get("detector", {}).get("error_type") in actual
                wrong_attempts += not before_correct
                repair_successes += not before_correct and after_correct
            key = ("correct_to_correct" if before_correct and after_correct else
                   "correct_to_wrong" if before_correct else
                   "wrong_to_correct" if after_correct else "wrong_to_wrong")
            transitions[key] += 1
        end_to_end = sorted(float(row.get("original_latency_seconds", 0)) +
                            float(row.get("correction_seconds", 0)) for row in predictions)
        p95 = end_to_end[min(len(end_to_end) - 1, int(.95 * len(end_to_end)))]
        return {"control_run_id": control_report["run_id"],
                "control_capture_sha256": hashlib.sha256((control_dir / "predictions.jsonl").read_bytes()).hexdigest(),
                "enabled_families": sorted(enabled), "transitions": transitions,
                "detection_precision": hits / detected if detected else None,
                "repair_success": repair_successes / wrong_attempts if wrong_attempts else None,
                "attempted_corrections": detected, "wrong_attempts": wrong_attempts,
                "net_corrections": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
                "end_to_end_p95_seconds": p95, "database_hashes_before": before,
                "database_hash_mismatches": changed_databases(before), "api_cost_usd": 0,
                "locked_final_opened": False}

    report = run_suite(run_cases, predict, scorer.metrics(include_ir=False), root / "v10/sql-agent/eval/pilot/deterministic",
        identity={"mode": "live", "capture_type": "deterministic_replay",
                  "candidate": "v10_b_deterministic", "version": REPLAY_VERSION,
                  "control_run_id": control_report["run_id"], "detector_calibration": calibration["version"],
                  "memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
                  "execution_policy": control_report["identity"]["execution_policy"], "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"], "correction_attempted": prediction.get("correction_attempted", False)})
    print(report["run_id"], report["metrics"], report["details"], flush=True)


if __name__ == "__main__":
    main()
