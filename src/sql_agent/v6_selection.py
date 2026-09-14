"""Development-only V6 model gates and final selection."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals import load_cases
from llm_evals.campaign import immutable_json

from .v3_metrics import Scoring
from .v6_campaign import MODES


def _run(root: Path, stage: str, mode: str) -> tuple[dict, list[dict]]:
    folder = root / "v6/sql-agent/eval" / stage / mode
    latest = json.loads((folder / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run = folder / latest
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if report["capture_status"] != "complete":
        raise ValueError(f"Incomplete capture: {stage}/{mode}")
    return report, predictions


def _candidate_oracle(cases, predictions, scorer):
    selected_correct = oracle_correct = 0
    for case, prediction in zip(cases, predictions):
        selected_correct += scorer.correct(case, prediction)
        attempts = [{**attempt.get("result", {}), "status": "completed"}
                    for attempt in prediction.get("attempts", []) if attempt.get("result")]
        oracle_correct += any(scorer.correct(case, attempt) for attempt in attempts)
    return selected_correct / len(cases), oracle_correct / len(cases)


def pilot_gate(root: Path) -> dict:
    cases = load_cases(root / "v6/sql-agent/data/development.jsonl")
    from .v6_campaign import _pilot
    cases = _pilot(cases)
    scorer = Scoring(cases)
    qwen, _ = _run(root, "pilot", "qwen_v5")
    arctic_rows = []
    for mode in ["arctic_direct", "arctic_plan"]:
        report, predictions = _run(root, "pilot", mode)
        selected, oracle = _candidate_oracle(cases, predictions, scorer)
        arctic_rows.append({"mode": mode, "execution_accuracy": report["metrics"]["execution_accuracy_v3"],
                            "completion": report["metrics"]["coverage"], "p95_seconds": report["p95_seconds"],
                            "candidate_oracle": oracle, "selected_accuracy": selected})
    best = max(arctic_rows, key=lambda row: (row["execution_accuracy"], -row["p95_seconds"]))
    qwen_ex = qwen["metrics"]["execution_accuracy_v3"]
    qualified = best["completion"] >= .95 and (best["execution_accuracy"] > qwen_ex or
                                                best["candidate_oracle"] - best["execution_accuracy"] >= .05)
    return {"version": "sql-v6-pilot-gate-v1", "qwen_execution_accuracy": qwen_ex, "arctic": arctic_rows,
            "best_arctic": best["mode"], "arctic_qualified": qualified,
            "xiyan_required": best["execution_accuracy"] < .65 or best["execution_accuracy"] - qwen_ex < .03}


def freeze_selection(root: Path) -> dict:
    cases = load_cases(root / "v6/sql-agent/data/development.jsonl")
    scorer = Scoring(cases)
    available = []
    for mode in MODES:
        try:
            report, predictions = _run(root, "development", mode)
        except FileNotFoundError:
            continue
        selected, oracle = _candidate_oracle(cases, predictions, scorer)
        available.append({"mode": mode, "execution_accuracy": report["metrics"]["execution_accuracy_v3"],
                          "completion": report["metrics"]["coverage"], "p95_seconds": report["p95_seconds"],
                          "selected_accuracy": selected, "candidate_oracle": oracle,
                          "oracle_gain": oracle - selected})
    if not any(row["mode"] == "qwen_v5" for row in available):
        raise ValueError("Development selection requires the Qwen V5 control")
    eligible = [row for row in available if row["completion"] >= .95 and row["p95_seconds"] <= 20]
    if not eligible:
        raise ValueError("No development candidate meets completion and latency gates")
    for row in eligible:
        if MODES[row["mode"]]["strategy"] == "adaptive_two":
            single = max((other for other in available if other["mode"].startswith(row["mode"].split("_")[0]) and
                          MODES[other["mode"]]["strategy"] == "single"),
                         key=lambda other: other["execution_accuracy"], default=None)
            row["adaptive_eligible"] = bool(single and row["candidate_oracle"] - single["execution_accuracy"] >= .05 and
                row["execution_accuracy"] - single["execution_accuracy"] >=
                .5 * (row["candidate_oracle"] - single["execution_accuracy"]))
        else:
            row["adaptive_eligible"] = True
    eligible = [row for row in eligible if row["adaptive_eligible"]]
    winner = max(eligible, key=lambda row: (row["execution_accuracy"], -row["p95_seconds"]))
    selection = {"version": "sql-v6-selection-v2", "winner": winner["mode"],
                 "configuration": MODES[winner["mode"]], "development_candidates": available,
                 "final_modes": ["qwen_v5", winner["mode"]],
                 "rule": "highest development EX after completion, latency, adaptive-oracle, and two-candidate gates",
                 "supersedes": "selection.json, whose Qwen control allowed three generations in two cases"}
    immutable_json(root / "v6/sql-agent/selection-v2.json", selection)
    return selection
