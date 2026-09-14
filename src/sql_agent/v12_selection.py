"""Immutable V12/V13 development selection and locked-final promotion decisions."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals.campaign import immutable_json

from .v8_selection import _interval, _run
from .v12_campaign import ARMS, decision_path, open_cases


# V13 reuses the V12 development control: same cases, code path, model digest, runtime, and decoding.
DEVELOPMENT_CONTROL = {"v12": "v12/sql-agent/eval/development/control",
                       "v13": "v12/sql-agent/eval/development/control"}


def _summary(report: dict) -> dict:
    return {"run_id": report["run_id"], "execution_accuracy": float(report["metrics"]["execution_accuracy_v7"]),
            "completion": float(report["metrics"]["completion"]), "p95_seconds": float(report["p95_seconds"]),
            "database_hash_mismatches": report.get("details", {}).get("database_hash_mismatches"),
            "official_gold_self_consistency": report["metrics"].get("official_gold_self_consistency")}


def freeze_development(root: Path, version: str = "v12") -> dict:
    root = root.resolve()
    target = root / decision_path(version)
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    cases = open_cases(root, "development", version)
    control_report, control_outcomes = _run(root, root / DEVELOPMENT_CONTROL[version])
    control = {**_summary(control_report), "source": DEVELOPMENT_CONTROL[version]}
    candidates = {}
    for arm in (name for name in ARMS[version] if name != "control"):
        report, outcomes = _run(root, root / version / "sql-agent/eval/development" / arm)
        row = _summary(report)
        row["ex_lift"] = row["execution_accuracy"] - control["execution_accuracy"]
        row["comparison"] = _interval(cases, control_outcomes, outcomes, seed=20260914)
        row["eligible"] = bool(row["ex_lift"] >= .03 and row["completion"] >= control["completion"] - .02
                               and row["p95_seconds"] <= 2 * control["p95_seconds"]
                               and row["database_hash_mismatches"] == [])
        candidates[arm] = row
    eligible = [arm for arm, row in candidates.items() if row["eligible"]]
    winner = max(eligible, key=lambda arm: (candidates[arm]["execution_accuracy"], candidates[arm]["completion"],
                                            -candidates[arm]["p95_seconds"])) if eligible else None
    result = {"version": f"sql-{version}-development-decision-v1", "case_count": len(cases), "control": control,
              "candidates": candidates, "winner": winner, "final_eligible": winner is not None,
              "rule": "EX >= control + 0.03; completion >= control - 0.02; p95 <= 2x control; zero database changes",
              "locked_final_opened": False}
    immutable_json(target, result)
    return result


def freeze_final(root: Path, version: str = "v12") -> dict:
    root = root.resolve()
    target = root / version / "sql-agent/final-decision.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    decision = json.loads((root / decision_path(version)).read_text(encoding="utf-8"))
    if not decision.get("final_eligible"):
        raise RuntimeError(f"{version.upper()} development selection did not permit the locked final")
    winner = decision["winner"]
    cases = open_cases(root, "final", version)
    control_report, control_outcomes = _run(root, root / version / "sql-agent/eval/final/control")
    winner_report, winner_outcomes = _run(root, root / version / "sql-agent/eval/final" / winner)
    control, candidate = _summary(control_report), _summary(winner_report)
    interval = _interval(cases, control_outcomes, winner_outcomes, seed=20260914)
    promoted = bool(interval["delta"] >= .03 and interval["low"] > 0
                    and candidate["completion"] >= control["completion"] - .02
                    and candidate["p95_seconds"] <= 2 * control["p95_seconds"]
                    and candidate["database_hash_mismatches"] == [] and control["database_hash_mismatches"] == []
                    and (candidate["official_gold_self_consistency"] or 0) >= .99)
    result = {"version": f"sql-{version}-final-decision-v1", "case_count": len(cases), "winner": winner,
              "control": control, "candidate": candidate, "comparison": interval, "promoted": promoted,
              "retained_default": winner if promoted else "qwen_v5"}
    immutable_json(target, result)
    return result
