"""Scoring-only audit of the immutable V8 recoveries and regressions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm_evals import load_cases

from .v7_evaluation import V7Scoring, structural_error_families
from .v9_utilization import aggregate, analyze, sql_effects


AUDIT_VERSION = "sql-v9-transition-audit-v1"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_dir(root: Path, version: str, scope: str, condition: str, run_id: str) -> Path:
    return root / version / "sql-agent/eval" / scope / condition / run_id


def _changed_components(left: str | None, right: str | None) -> list[str]:
    try:
        before = Counter(row["kind"] for row in sql_effects(left or ""))
        after = Counter(row["kind"] for row in sql_effects(right or ""))
        return sorted(kind for kind in set(before) | set(after) if before[kind] != after[kind]) or ["expression_or_binding"]
    except Exception:
        return ["syntax_or_extraction"]


def build(root: Path) -> dict:
    root = root.resolve()
    target = root / "v9/sql-agent/audit"
    target.mkdir(parents=True, exist_ok=True)
    cases = load_cases(root / "v8/sql-agent/data/decomposition.jsonl")
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    v7 = json.loads((root / "v7/sql-agent/reproduction-decision.json").read_text(encoding="utf-8"))
    no_id = v7["conditions"]["reference_none"]["run_id"]
    no_rows = _jsonl(_run_dir(root, "v7", "reproduction", "reference_none", no_id) / "predictions.jsonl")
    latest = json.loads((root / "v8/sql-agent/eval/development/generated_all/latest.json").read_text(encoding="utf-8"))
    generated_rows = _jsonl(_run_dir(root, "v8", "development", "generated_all", latest["run_id"]) / "predictions.jsonl")
    no_by_id = {row["id"]: row for row in no_rows}
    generated_by_id = {row["id"]: row for row in generated_rows}
    rows, utilization = [], []
    for case in cases:
        no, generated = no_by_id[case.id], generated_by_id[case.id]
        left, right = scorer.correct(case, no), scorer.correct(case, generated)
        if left == right:
            continue
        report = analyze(case.expected["sql"], generated.get("sql"), generated.get("evidence_packet", []))
        utilization.append(report)
        label = "recovery" if right else "regression"
        relevant = report["relevant_fact_count"]
        correct_use = report["counts"].get("correctly_used", 0)
        rows.append({
            "version": AUDIT_VERSION, "id": case.id, "db_id": case.metadata["db_id"],
            "transition": label, "no_evidence_correct": left, "generated_evidence_correct": right,
            "evidence_fact_count": len(generated.get("evidence_packet", [])),
            "evidence_relevant": "yes" if relevant == len(generated.get("evidence_packet", [])) else
                                 "partly" if relevant else "no",
            "evidence_utilization": report,
            "sql_components_changed": _changed_components(no.get("sql"), generated.get("sql")),
            "generated_error_families": structural_error_families(case.expected["sql"], generated.get("sql")),
            "change_desirable": right,
            "sql_used_relevant_evidence": correct_use > 0,
            "gold_confined_to_scoring": True,
        })
    if Counter(row["transition"] for row in rows) != Counter({"recovery": 5, "regression": 3}):
        raise ValueError("V9 audit must reproduce immutable V8's five recoveries and three regressions")
    output = target / "transitions.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"version": AUDIT_VERSION, "cases": len(rows),
              "transitions": dict(Counter(row["transition"] for row in rows)),
              "changed_components": dict(Counter(value for row in rows for value in row["sql_components_changed"])),
              "utilization": aggregate(utilization), "gold_confined_to_scoring": True}
    (target / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
