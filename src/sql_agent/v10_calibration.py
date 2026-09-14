"""Scoring-only detector calibration on already inspected captures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm_evals.campaign import immutable_json

from .semantic_ir import reference_structure
from .v7_evaluation import V7Scoring
from .v8_data import open_cases, open_pilot
from .v8_selection import _run
from .v9_utilization import sql_effects
from .v10_correction import detect
from .v10_correction import localized_patch
from .execution import execute_sql
from .guardrails import validate


CALIBRATION_VERSION = "sql-v10-detector-calibration-v5"


def _effect_set(sql: str | None, kind: str):
    if not sql:
        return set()
    try:
        return {json.dumps(row, sort_keys=True) for row in sql_effects(sql) if row["kind"] == kind}
    except Exception:
        return set()


def actual_families(reference_sql: str, predicted_sql: str | None) -> set[str]:
    if not predicted_sql:
        return {"operation", "grain", "predicate", "join"}
    try:
        gold, candidate = reference_structure(reference_sql), reference_structure(predicted_sql)
    except Exception:
        return {"operation", "grain", "predicate", "join"}
    result = set()
    if gold["aggregates"] != candidate["aggregates"]:
        result.add("operation")
    if gold["grouping"] != candidate["grouping"] or (
            gold["aggregates"] and gold["projection_count"] != candidate["projection_count"]):
        result.add("grain")
    if _effect_set(reference_sql, "predicate") != _effect_set(predicted_sql, "predicate"):
        result.add("predicate")
    if gold["joins"] != candidate["joins"] or _effect_set(reference_sql, "join") != _effect_set(predicted_sql, "join"):
        result.add("join")
    return result


def calibrate(root: Path) -> dict:
    root = root.resolve(); target = root / "v10/sql-agent/detector-calibration-v5.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    v8_cases = open_cases(root)
    _, v8_predictions = _load_predictions(root / "v8/sql-agent/eval/development/generated_all")
    v9_cases = open_pilot(root)
    _, v9_predictions = _load_predictions(root / "v9/sql-agent/eval/pilot/scoped")
    # Prefer the V9 scoped capture for its 20 IDs; add the remaining inspected
    # V8 cases to obtain 40 calibration cases without duplicating outcomes.
    case_by_id = {case.id: case for case in v8_cases}
    predictions = {**v8_predictions, **v9_predictions}
    ids = [case.id for case in v8_cases]
    cases = [case_by_id[identifier] for identifier in ids]
    from .v10_data import open_memory_capture
    memory_cases = open_memory_capture(root)
    _, memory_predictions = _load_predictions(root / "v10/sql-agent/eval/memory/scoped_control")
    cases = [*cases, *memory_cases]
    predictions.update(memory_predictions)
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    rows, family_counts, correct_interventions = [], Counter(), 0
    for case in cases:
        prediction = predictions[case.id]
        detection = detect(case.input.split("Benchmark evidence:", 1)[0].strip(), prediction.get("sql", ""),
                           prediction.get("evidence_packet", []), prediction)
        actual = actual_families(case.expected["sql"], prediction.get("sql"))
        correct_before = scorer.correct(case, prediction)
        hit = bool(detection.error_type and detection.error_type in actual)
        repair_correct = None; repair_transition = "not_attempted"
        patch = localized_patch(prediction.get("sql", ""), detection)
        if patch:
            try:
                sql = validate(patch["sql"])
                result = execute_sql(Path(case.context["database"]), sql, timeout=10)
                repaired = {**result, "status": "completed", "sql": sql}
                repair_correct = scorer.correct(case, repaired)
                repair_transition = ("correct_to_correct" if correct_before and repair_correct else
                                     "correct_to_wrong" if correct_before else
                                     "wrong_to_correct" if repair_correct else "wrong_to_wrong")
            except (ValueError, RuntimeError, TimeoutError):
                repair_correct = False
                repair_transition = "repair_failed"
        if detection.error_type:
            family_counts[(detection.error_type, "flagged")] += 1
            family_counts[(detection.error_type, "correct")] += hit
            family_counts[(detection.error_type, "repair_success")] += repair_transition == "wrong_to_correct"
            family_counts[(detection.error_type, "wrong_attempts")] += not correct_before
            family_counts[(detection.error_type, "regression")] += repair_transition == "correct_to_wrong"
            correct_interventions += correct_before
        rows.append({"id": case.id, "detected": detection.model_dump(), "actual_families": sorted(actual),
                     "detector_hit": hit, "correct_query_flagged": bool(detection.error_type and correct_before),
                     "repair_transition": repair_transition, "repair_correct": repair_correct})
    families = {}
    for family in ["operation", "grain", "predicate", "join"]:
        flagged, correct = family_counts[(family, "flagged")], family_counts[(family, "correct")]
        precision = correct / flagged if flagged else None
        wrong_attempts = family_counts[(family, "wrong_attempts")]
        repair_success = family_counts[(family, "repair_success")] / wrong_attempts if wrong_attempts else None
        regressions = family_counts[(family, "regression")]
        families[family] = {"flagged": flagged, "correct": correct, "precision": precision,
                            "wrong_attempts": wrong_attempts, "repair_success": repair_success,
                            "correct_query_regressions": regressions,
                            "enabled": bool(flagged >= 2 and precision is not None and precision >= .80 and
                                            repair_success is not None and repair_success >= .50 and regressions == 0)}
    flagged_total = sum(value["flagged"] for value in families.values())
    correct_total = sum(value["correct"] for value in families.values())
    report = {"version": CALIBRATION_VERSION, "cases": len(cases), "families": families,
              "overall_precision": correct_total / flagged_total if flagged_total else None,
              "flagged": flagged_total, "correct_query_interventions": correct_interventions,
              "enabled_families": [name for name, value in families.items() if value["enabled"]],
              "minimum_precision": .80, "minimum_family_support": 2,
              "source": "40 inspected cases plus 30 BIRD-training natural-error captures; V9 scoped preferred for overlaps",
              "gold_confined_to_scoring": True, "locked_final_opened": False}
    detail_path = target.with_name("detector-calibration-cases.jsonl")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    immutable_json(target, report); return report


def _load_predictions(folder: Path):
    report, _ = _run(folder.parents[4], folder)  # root argument is unused by _run
    path = folder / report["run_id"] / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return report, {row["id"]: row for row in rows}
