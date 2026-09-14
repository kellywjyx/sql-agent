"""Final V6 paired comparison and promotion report."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals import load_cases
from llm_evals.campaign import grouped_comparison, immutable_json

from .v3_metrics import Scoring
from .v6_selection import _run


def final_report(root: Path) -> dict:
    selection = json.loads((root / "v6/sql-agent/selection.json").read_text(encoding="utf-8"))
    winner = selection["winner"]
    cases = load_cases(root / "v6/sql-agent/data/final.jsonl")
    control_report, control_predictions = _run(root, "final", "qwen_v5")
    candidate_report, candidate_predictions = _run(root, "final", winner)
    scorer = Scoring(cases)
    metric = scorer.metrics()[0]
    comparison = grouped_comparison(metric, cases, control_predictions, candidate_predictions,
                                    group_key="bootstrap_group", samples=2000, seed=20260903)
    candidate_ex = candidate_report["metrics"]["execution_accuracy_v3"]
    control_ex = control_report["metrics"]["execution_accuracy_v3"]
    mismatches = candidate_report.get("details", {}).get("database_hash_mismatches", [])
    reasons = []
    if candidate_ex < .65: reasons.append("final EX below 0.65")
    if candidate_ex - control_ex < .05: reasons.append("improvement below five points")
    if comparison["low"] <= 0: reasons.append("paired interval is not positive")
    if candidate_report["metrics"]["coverage"] < .97: reasons.append("completion below 0.97")
    if candidate_report["p95_seconds"] > 20: reasons.append("p95 exceeds 20 seconds")
    if candidate_report["p95_seconds"] > 2 * control_report["p95_seconds"]: reasons.append("p95 exceeds twice control")
    if mismatches: reasons.append("database bytes changed")
    report = {"version": "sql-v6-final-report-v1", "winner": winner, "dataset_count": len(cases),
              "control": {"execution_accuracy": control_ex, "completion": control_report["metrics"]["coverage"],
                          "p95_seconds": control_report["p95_seconds"]},
              "candidate": {"execution_accuracy": candidate_ex, "completion": candidate_report["metrics"]["coverage"],
                            "p95_seconds": candidate_report["p95_seconds"]},
              "comparison": comparison, "promoted": not reasons, "reasons": reasons,
              "api_charge_usd": 0, "scope": "BIRD development complement; not a leaderboard submission"}
    immutable_json(root / "v6/sql-agent/final-report.json", report)
    return report
