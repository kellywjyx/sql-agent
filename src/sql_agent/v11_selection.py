"""Immutable V11 pilot, development, and regression gates."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals.campaign import immutable_json

from .v8_selection import _interval, _run
from .v11_data import open_role


def _value(report: dict) -> tuple[float, float, float]:
    return (float(report["metrics"]["execution_accuracy_v7"]),
            float(report["metrics"]["completion"]), float(report["p95_seconds"]))


def freeze_pilot(root: Path) -> dict:
    root = root.resolve(); target = root / "v11/sql-agent/pilot-decision.json"
    if target.exists(): return json.loads(target.read_text(encoding="utf-8"))
    cases = open_role(root, "development")[:60]; ids = [case.id for case in cases]
    base_report, base_all = _run(root, root / "v11/sql-agent/eval/pilot/base")
    base = {identifier: base_all[identifier] for identifier in ids}; base_ex, _, _ = _value(base_report)
    candidates = {}
    for recipe in ("A", "B"):
        report, outcomes = _run(root, root / f"v11/sql-agent/eval/pilot/{recipe}")
        aligned = {identifier: outcomes[identifier] for identifier in ids}
        ex, completion, p95 = _value(report)
        training = json.loads((root / f"v11/sql-agent/training/pilot/{recipe}/training.json").read_text(encoding="utf-8"))
        reload_path = root / f"v11/sql-agent/training/pilot/{recipe}/reload.json"
        eligible = bool(ex - base_ex >= .05 and completion >= .95 and
                        not report["details"]["database_hash_mismatches"] and reload_path.exists())
        candidates[recipe] = {"run_id": report["run_id"], "execution_accuracy": ex,
            "ex_lift": ex - base_ex, "completion": completion, "p95_seconds": p95,
            "validation_loss": min((row["eval_loss"] for row in training["loss_history"] if "eval_loss" in row), default=None),
            "finite_loss": True, "reload_verified": reload_path.exists(), "eligible": eligible,
            "comparison": _interval(cases, base, aligned, seed=20260923),
            "database_hash_mismatches": report["details"]["database_hash_mismatches"]}
    eligible = [name for name, value in candidates.items() if value["eligible"]]
    selected = max(eligible, key=lambda name: (candidates[name]["execution_accuracy"],
        candidates[name]["completion"], -(candidates[name]["validation_loss"] or 999),
        -candidates[name]["p95_seconds"])) if eligible else None
    result = {"version": "sql-v11-pilot-decision-v1", "case_count": len(cases),
        "base": {"run_id": base_report["run_id"], "execution_accuracy": base_ex,
                 "completion": base_report["metrics"]["completion"], "p95_seconds": base_report["p95_seconds"]},
        "candidates": candidates, "selected_recipe": selected,
        "full_training_eligible": selected is not None, "locked_final_opened": False, "api_cost_usd": 0}
    immutable_json(target, result); return result


def freeze_development(root: Path) -> dict:
    root = root.resolve(); target = root / "v11/sql-agent/development-decision.json"
    if target.exists(): return json.loads(target.read_text(encoding="utf-8"))
    pilot = json.loads((root / "v11/sql-agent/pilot-decision.json").read_text(encoding="utf-8"))
    if not pilot.get("full_training_eligible"): raise RuntimeError("V11 pilot did not permit full training")
    cases = open_role(root, "development"); ids = [case.id for case in cases]
    base_report, base_all = _run(root, root / "v11/sql-agent/eval/development/base")
    adapted_report, adapted_all = _run(root, root / "v11/sql-agent/eval/development/selected")
    base = {identifier: base_all[identifier] for identifier in ids}
    adapted = {identifier: adapted_all[identifier] for identifier in ids}
    base_ex, _, _ = _value(base_report); adapted_ex, completion, p95 = _value(adapted_report)
    eligible = bool(adapted_ex >= .55 and adapted_ex - base_ex >= .08 and completion >= .95 and
                    not adapted_report["details"]["database_hash_mismatches"])
    result = {"version": "sql-v11-development-decision-v1", "selected_recipe": pilot["selected_recipe"],
        "base_ex": base_ex, "adapter_ex": adapted_ex, "ex_lift": adapted_ex - base_ex,
        "completion": completion, "p95_seconds": p95,
        "comparison": _interval(cases, base, adapted, seed=20260923),
        "regression_eligible": eligible, "locked_final_opened": False, "api_cost_usd": 0}
    immutable_json(target, result); return result


def freeze_regression(root: Path) -> dict:
    root = root.resolve(); target = root / "v11/sql-agent/regression-decision.json"
    if target.exists(): return json.loads(target.read_text(encoding="utf-8"))
    development = json.loads((root / "v11/sql-agent/development-decision.json").read_text(encoding="utf-8"))
    if not development.get("regression_eligible"): raise RuntimeError("V11 development gate did not permit regression")
    report, _ = _run(root, root / "v11/sql-agent/eval/regression/selected")
    ex, completion, p95 = _value(report); delta = ex - .545
    # The historical paired interval is produced only when the original V5
    # case outcomes can be aligned; promotion remains false without it.
    interval = report.get("details", {}).get("comparison_vs_v5")
    promoted = bool(ex >= .60 and delta >= .05 and interval and interval["low"] > 0 and
                    completion >= .97 and p95 < 20 and not report["details"]["database_hash_mismatches"])
    result = {"version": "sql-v11-regression-decision-v1", "execution_accuracy": ex,
        "delta_vs_recorded_v5": delta, "completion": completion, "p95_seconds": p95,
        "comparison_vs_v5": interval, "promoted": promoted,
        "retained_public_default": "v11_adapter" if promoted else "qwen_v5",
        "locked_final_opened": False, "locked_final_requires_separate_approval": True, "api_cost_usd": 0}
    immutable_json(target, result); return result
