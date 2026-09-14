"""Scoring-only V8 evidence and oracle-gap metrics."""
from __future__ import annotations

import json
from pathlib import Path

from .value_index import normalize_value
from .v8_decomposition import _reference_targets


METRIC_VERSION = "sql-v8-evidence-metrics-v1"


def _target_sets(case) -> dict:
    targets = _reference_targets(case.expected["sql"])
    columns = {(str(row.get("table") or "").casefold(), str(row["column"]).casefold())
               for row in targets["columns"]}
    literals = {(str(row.get("table") or "").casefold(), str(row["column"]).casefold(),
                 normalize_value(row["value"])) for row in targets["literals"]}
    return {"tables": {str(value).casefold() for value in targets["tables"]},
            "columns": columns, "literals": literals,
            "transformations": set(targets["transformations"])}


def evidence_metrics(cases, packet_rows: list[dict], decomposition_rows: list[dict] | None = None) -> dict:
    if [case.id for case in cases] != [row["id"] for row in packet_rows]:
        raise ValueError("V8 evidence scoring requires aligned unique case IDs")
    groups = {row["id"]: row["group"] for row in (decomposition_rows or [])}
    required_literals = recovered_literals = 0
    supplied = relevant = mapped = mapping_candidates = 0
    required_transforms = recovered_transforms = 0
    table_scores, column_scores = [], []
    by_group = {}
    for case, row in zip(cases, packet_rows):
        target = _target_sets(case)
        facts = row.get("facts", [])
        predicted_tables = {str(fact.get("table", "")).casefold() for fact in facts}
        predicted_columns = {(str(fact.get("table", "")).casefold(), str(fact.get("column", "")).casefold())
                             for fact in facts if fact.get("column")}
        predicted_literals = {(str(fact.get("table", "")).casefold(),
                               str(fact.get("column", "")).casefold(), normalize_value(fact.get("db_value", "")))
                              for fact in facts if fact.get("db_value") is not None}
        predicted_transforms = {str(fact.get("transformation")) for fact in facts if fact.get("transformation")}
        if target["tables"]:
            table_scores.append(len(target["tables"] & predicted_tables) / len(target["tables"]))
        if target["columns"]:
            # Unqualified gold columns match the same column name in any predicted table.
            hits = sum(any(pc == column and (not table or pt == table) for pt, pc in predicted_columns)
                       for table, column in target["columns"])
            column_scores.append(hits / len(target["columns"]))
        literal_hits = sum(any(pv == value and pc == column and (not table or pt == table)
                               for pt, pc, pv in predicted_literals)
                           for table, column, value in target["literals"])
        required_literals += len(target["literals"]); recovered_literals += literal_hits
        for fact in facts:
            if fact.get("type") not in {"literal_value", "encoded_value", "normalization"}:
                continue
            supplied += 1
            candidate = (str(fact.get("table", "")).casefold(), str(fact.get("column", "")).casefold(),
                         normalize_value(fact.get("db_value", "")))
            target_values = {value for _, _, value in target["literals"]}
            mapping_candidates += candidate[2] in target_values
            is_relevant = any(candidate[2] == value and candidate[1] == column and
                              (not table or candidate[0] == table) for table, column, value in target["literals"])
            relevant += is_relevant
            mapped += is_relevant
        required_transforms += len(target["transformations"])
        recovered_transforms += len(target["transformations"] & predicted_transforms)
        group = groups.get(case.id, "all")
        item = by_group.setdefault(group, {"cases": 0, "required_literals": 0, "literal_hits": 0})
        item["cases"] += 1; item["required_literals"] += len(target["literals"]); item["literal_hits"] += literal_hits
    average = lambda values: sum(values) / len(values) if values else 0.0
    for value in by_group.values():
        value["literal_recall"] = value["literal_hits"] / value["required_literals"] if value["required_literals"] else 0.0
    return {"version": METRIC_VERSION, "cases": len(cases),
            "table_recall": average(table_scores), "column_recall": average(column_scores),
            "literal_recall": recovered_literals / required_literals if required_literals else 0.0,
            "required_literals": required_literals, "recovered_literals": recovered_literals,
            "evidence_precision": relevant / supplied if supplied else 0.0,
            "supplied_literal_facts": supplied, "relevant_literal_facts": relevant,
            "value_column_accuracy": mapped / mapping_candidates if mapping_candidates else 0.0,
            "value_mapping_candidates": mapping_candidates,
            "transformation_recall": recovered_transforms / required_transforms if required_transforms else 0.0,
            "by_group": by_group}


def oracle_gap_recovery(no_evidence_ex: float, oracle_ex: float, generated_ex: float) -> float | None:
    gap = oracle_ex - no_evidence_ex
    return (generated_ex - no_evidence_ex) / gap if gap > 0 else None


def transitions(no_correct: dict[str, bool], generated_correct: dict[str, bool]) -> dict:
    if set(no_correct) != set(generated_correct):
        raise ValueError("Transition analysis requires identical case IDs")
    result = {"both_correct": 0, "wrong_to_correct": 0, "correct_to_wrong": 0, "both_wrong": 0}
    for identifier in no_correct:
        left, right = no_correct[identifier], generated_correct[identifier]
        key = "both_correct" if left and right else "correct_to_wrong" if left else "wrong_to_correct" if right else "both_wrong"
        result[key] += 1
    return result


def load_decomposition(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
