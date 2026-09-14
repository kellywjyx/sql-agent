"""Freeze answer-aware V7 development selection outside the application boundary."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals.campaign import immutable_json

from .v7_critic import calibrate_rules
from .v7_data import open_cases
from .v7_evaluation import V7Scoring


def _latest(root: Path, stage: str, mode: str):
    folder = root / f"v7/sql-agent/eval/{stage}/{mode}"
    run_id = json.loads((folder / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run = folder / run_id
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if report.get("capture_status") != "complete":
        raise ValueError(f"Incomplete V7 capture: {stage}/{mode}")
    return report, predictions


def freeze_pilot_decision(root: Path) -> dict:
    """Record the mandatory 20-case stop/continue decision without opening later roles."""
    root = root.resolve()
    target = root / "v7/sql-agent/pilot-decision.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    report, predictions = _latest(root, "selection", "v7_three")
    if report.get("count") != 20 or len(predictions) != 20:
        raise ValueError("V7 pilot decision requires exactly 20 captured selection cases")
    metrics, details = report["metrics"], report.get("details", {})
    reasons = []
    if metrics["candidate_oracle_ex"] < .65:
        reasons.append("candidate oracle EX below the 0.65 pilot continuation gate")
    if metrics["completion"] < .97:
        reasons.append("completion below 0.97")
    if details.get("database_hash_mismatches"):
        reasons.append("database bytes changed")
    path_counts: dict[str, int] = {}
    for prediction in predictions:
        for candidate in prediction.get("candidates", []):
            path = candidate.get("path", "unknown")
            path_counts[path] = path_counts.get(path, 0) + 1
    decision = {
        "version": "sql-v7-pilot-decision-v1",
        "continue_full_selection": not reasons,
        "reasons": reasons,
        "metrics": {
            "selected_ex": metrics["execution_accuracy_v7"],
            "first_candidate_ex": metrics["first_candidate_accuracy"],
            "candidate_oracle_ex": metrics["candidate_oracle_ex"],
            "completion": metrics["completion"],
            "p95_seconds": report["p95_seconds"],
            "table_recall": details.get("evidence_table_recall"),
            "column_recall": details.get("evidence_column_recall"),
            "literal_recall": details.get("evidence_literal_recall"),
        },
        "path_capture_counts": path_counts,
        "database_hash_mismatches": details.get("database_hash_mismatches", []),
        "locked_final_opened": False,
        "retained_default": "qwen_v5",
        "run_id": report["run_id"],
    }
    immutable_json(target, decision)
    return decision


def freeze_reproduction_decision(root: Path) -> dict:
    """Record prompt/evidence ablations and the separately approved Q5 eligibility gate."""
    root = root.resolve()
    target = root / "v7/sql-agent/reproduction-decision.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    modes = ["v6_arctic_oracle", "reference_oracle", "reference_none", "reference_generated"]
    summaries = {}
    for mode in modes:
        report, predictions = _latest(root, "reproduction", mode)
        if report.get("count") != 40 or len(predictions) != 40:
            raise ValueError(f"Reproduction decision requires 40 cases for {mode}")
        summaries[mode] = {
            "run_id": report["run_id"],
            "execution_accuracy": report["metrics"]["execution_accuracy_v7"],
            "completion": report["metrics"]["completion"],
            "p95_seconds": report["p95_seconds"],
            "official_scorer_agreement": report["metrics"]["scorer_agreement"],
            "official_gold_self_consistency": report["metrics"]["official_gold_self_consistency"],
            "literal_recall": report.get("details", {}).get("evidence_literal_recall"),
            "database_hash_mismatches": report.get("details", {}).get("database_hash_mismatches", []),
        }
    published = .689
    local_reference = summaries["reference_oracle"]["execution_accuracy"]
    decision = {
        "version": "sql-v7-reproduction-decision-v1",
        "conditions": summaries,
        "published_bf16_model_card_result": published,
        "local_q4_reference_oracle_ex": local_reference,
        "published_gap": published - local_reference,
        "q5_matched_pilot_eligible": published - local_reference >= .10,
        "q5_requires_separate_download_approval": True,
        "q5_downloaded": False,
        "conclusion": "Prompt serialization improved completion but not EX; oracle evidence remains materially useful.",
        "locked_final_opened": False,
    }
    immutable_json(target, decision)
    return decision


def freeze(root: Path) -> dict:
    root = root.resolve()
    target = root / "v7/sql-agent/selection.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    calibration_cases = open_cases(root, "calibration")
    calibration_report, calibration_predictions = _latest(root, "calibration", "v7_three")
    if calibration_report.get("count") != len(calibration_cases) or len(calibration_predictions) != len(calibration_cases):
        raise ValueError("Full 60-case critic calibration is required before V7 selection")
    calibration_scorer = V7Scoring(calibration_cases)
    critic_reports, critic_correct = [], []
    for case, prediction in zip(calibration_cases, calibration_predictions):
        for candidate in prediction.get("candidates", []):
            if candidate.get("critic"):
                critic_reports.append(candidate["critic"])
                critic_correct.append(calibration_scorer.candidate_correct(case, candidate))
    calibration = calibrate_rules(critic_reports, critic_correct, minimum_precision=.90)
    flagged = sum(row["flagged"] for row in calibration["rules"].values())
    true_flags = sum(row["true_flags"] for row in calibration["rules"].values())
    aggregate_precision = true_flags / flagged if flagged else 0
    calibration["aggregate_flag_precision"] = aggregate_precision
    immutable_json(root / "v7/sql-agent/critic-calibration.json", calibration)

    selection_cases = open_cases(root, "selection")
    report, predictions = _latest(root, "selection", "v7_three")
    if report.get("count") != len(selection_cases) or len(predictions) != len(selection_cases):
        raise ValueError("Full 100-case selection is required before freezing V7")
    scorer = V7Scoring(selection_cases)
    path_correct: dict[str, set[str]] = {}
    for case, prediction in zip(selection_cases, predictions):
        for candidate in prediction.get("candidates", []):
            if scorer.candidate_correct(case, candidate):
                path_correct.setdefault(candidate.get("path", "unknown"), set()).add(case.id)
    union = set().union(*path_correct.values()) if path_correct else set()
    metrics = report["metrics"]
    reasons = []
    if metrics["candidate_oracle_ex"] < .70:
        reasons.append("candidate oracle EX below 0.70")
    if metrics["execution_accuracy_v7"] < .60:
        reasons.append("selected EX below 0.60")
    if metrics["completion"] < .97:
        reasons.append("completion below 0.97")
    if aggregate_precision < .90:
        reasons.append("aggregate critic flag precision below 0.90")
    if report.get("details", {}).get("database_hash_mismatches"):
        reasons.append("database bytes changed")
    selection = {"version": "sql-v7-selection-v1", "eligible_for_regression": not reasons,
        "reasons": reasons, "research_configuration": {"candidate_paths": ["arctic_direct", "arctic_ir", "qwen_decomposed"],
            "evidence_mode": "generated", "candidate_limit": 3},
        "interactive_configuration": {"pipeline_profile": "v7_grounded", "schema_mode": "auto",
            "candidate_paths": ["arctic_direct", "arctic_ir"], "candidate_limit": 2},
        "metrics": {"selected_ex": metrics["execution_accuracy_v7"], "candidate_oracle_ex": metrics["candidate_oracle_ex"],
            "completion": metrics["completion"], "p95_seconds": report["p95_seconds"],
            "critic_aggregate_precision": aggregate_precision},
        "candidate_complementarity": {"path_correct_counts": {key: len(value) for key, value in path_correct.items()},
            "oracle_union_count": len(union), "neither_correct_count": len(selection_cases) - len(union)},
        "critic_calibration_identity": "critic-calibration.json",
        "locked_final_opened": False,
        "rule": "oracle>=0.70, selected>=0.60, completion>=0.97, critic precision>=0.90, unchanged DB bytes"}
    immutable_json(target, selection)
    return selection
