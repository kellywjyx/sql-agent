"""Calibratable intent critic, result evidence, voting, and local SQL repairs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import sqlglot
from sqlglot import exp

from .semantic_ir import GroundedIntent


CRITIC_VERSION = "intent-critic-v2"
RESULT_SIGNAL_VERSION = "sql-result-signals-v1"
REPAIR_VERSION = "sql-localized-repair-v1"


def _json_value(value):
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, float):
        return round(value, 12)
    return value


def result_signals(result: dict) -> dict:
    rows = [[_json_value(value) for value in row] for row in result.get("rows", [])]
    row_keys = [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows]
    unordered = sorted(row_keys)
    values = [value for row in result.get("rows", []) for value in row]
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    nulls = sum(value is None for value in values)
    payload = {
        "version": RESULT_SIGNAL_VERSION,
        "row_count": int(result.get("row_count", len(rows))),
        "column_count": len(result.get("columns", [])),
        "null_proportion": nulls / len(values) if values else 0.0,
        "duplicate_rate": 1 - len(set(row_keys)) / len(row_keys) if row_keys else 0.0,
        "numeric_min": min(numeric) if numeric else None,
        "numeric_max": max(numeric) if numeric else None,
        "sample_rows": rows[:3],
        "join_cardinality_changes": result.get("join_cardinality_changes", []),
        "ordered_hash": hashlib.sha256("\n".join(row_keys).encode()).hexdigest(),
        "unordered_hash": hashlib.sha256("\n".join(unordered).encode()).hexdigest(),
    }
    return payload


def structural_signature(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    selects = list(tree.find_all(exp.Select))
    top = selects[0] if selects else None
    order = next(tree.find_all(exp.Order), None)
    predicates = []
    for kind in (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Between):
        predicates.extend(node.sql(dialect="sqlite", normalize=True) for node in tree.find_all(kind))
    joins = []
    for join in tree.find_all(exp.Join):
        joins.append({"table": join.this.sql(dialect="sqlite", normalize=True),
                      "on": (join.args.get("on").sql(dialect="sqlite", normalize=True)
                             if join.args.get("on") else None)})
    return {
        "tables": sorted({node.name.casefold() for node in tree.find_all(exp.Table)}),
        "columns": sorted({node.name.casefold() for node in tree.find_all(exp.Column)}),
        "projection_count": len(top.expressions) if top else 0,
        "projections": sorted(expression.sql(dialect="sqlite", normalize=True) for expression in (top.expressions if top else [])),
        "predicates": sorted(predicates), "joins": joins,
        "aggregates": sorted(type(node).__name__.casefold() for node in tree.find_all(exp.AggFunc)),
        "grouping": sorted(node.sql(dialect="sqlite", normalize=True)
                           for group in tree.find_all(exp.Group) for node in group.expressions),
        "having": bool(next(tree.find_all(exp.Having), None)),
        "distinct": bool(top and top.args.get("distinct")),
        "ordering": order.sql(dialect="sqlite", normalize=True) if order else None,
        "limit": (top.args.get("limit").sql(dialect="sqlite", normalize=True)
                  if top and top.args.get("limit") else None),
        "subqueries": len(list(tree.find_all(exp.Subquery))),
        "set_operation": type(tree).__name__.casefold() if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)) else "none",
    }


def diversity(left_sql: str, right_sql: str) -> dict:
    left, right = structural_signature(left_sql), structural_signature(right_sql)
    dimensions = [name for name in ["tables", "columns", "projection_count", "predicates", "joins", "aggregates",
                                         "grouping", "having", "distinct", "ordering", "limit", "subqueries", "set_operation"]
                  if left[name] != right[name]]
    return {"different": bool(dimensions), "dimensions": dimensions,
            "left": left, "right": right}


def _aggregate_names(intent: GroundedIntent) -> set[str]:
    mapping = {"count": "count", "sum": "sum", "avg": "avg", "min": "min", "max": "max"}
    return {mapping[row.aggregation] for row in intent.semantic_intent.measures if row.aggregation in mapping}


def intent_critic(sql: str, grounded: GroundedIntent, schema: list[dict], evidence_facts: list[dict],
                  result: dict | None = None, *, calibrated_rules: set[str] | None = None) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    signature = structural_signature(sql)
    intent = grounded.semantic_intent
    high_bindings = {concept: rows[0] for concept, rows in grounded.bindings.items()
                     if rows and rows[0].confidence == "high"}
    actual_columns = {node.name.casefold() for node in tree.find_all(exp.Column)}
    actual_tables = set(signature["tables"])
    projection_nodes = list(tree.expressions) if isinstance(tree, exp.Select) else []
    projection_columns = {node.name.casefold() for expression in projection_nodes for node in expression.find_all(exp.Column)}
    where_columns = {node.name.casefold() for where in tree.find_all(exp.Where) for node in where.find_all(exp.Column)}
    required_projection = {high_bindings[row.concept].column.casefold() for row in intent.projections
                           if row.concept in high_bindings and high_bindings[row.concept].column}
    required_filters = {high_bindings[row.concept].column.casefold() for row in intent.filters
                        if row.concept in high_bindings and high_bindings[row.concept].column}
    expected_literals = [str(row.value) for row in [*intent.filters, *intent.having] if row.value is not None]
    literals = {str(node.this).casefold() for node in tree.find_all(exp.Literal)}
    expected_aggregates = _aggregate_names(grounded)
    actual_aggregates = set(signature["aggregates"])
    expected_projection_count = len(intent.projections) + len(intent.measures)
    result_data = result_signals(result) if result else None
    expected_grain_scalar = ("scalar" in intent.result_grain.casefold() or "entire" in intent.result_grain.casefold() or
                             "global" in intent.result_grain.casefold())
    expected_grain_grouped = bool(intent.grouping) or any(term in intent.result_grain.casefold()
                                                           for term in ["per ", "each ", "one row per"])
    checks = {
        "known_tables": actual_tables <= {table["table"].casefold() for table in schema},
        "required_tables": not grounded.high_confidence or {table.casefold() for table in grounded.selected_tables} <= actual_tables,
        "projection_count": expected_projection_count == 0 or len(projection_nodes) == expected_projection_count,
        "projection_bindings": not required_projection or required_projection <= projection_columns,
        "filter_bindings": not required_filters or required_filters <= where_columns,
        "literal_values": not expected_literals or all(value.casefold() in literals for value in expected_literals),
        "aggregate_functions": not expected_aggregates or expected_aggregates <= actual_aggregates,
        "grouping_grain": not expected_grain_grouped or bool(signature["grouping"]),
        "global_scalar_grain": not expected_grain_scalar or not signature["grouping"],
        "having": not intent.having or signature["having"],
        "distinct": not intent.distinct_required or signature["distinct"],
        "ordering": not intent.ordering or signature["ordering"] is not None,
        "ordering_direction": not intent.ordering or intent.ordering.direction == "unknown" or
                              f" {intent.ordering.direction}" in (signature["ordering"] or "").casefold(),
        "limit": intent.limit is None or signature["limit"] is not None,
        "unnecessary_limit": intent.limit is not None or signature["limit"] is None,
        "subquery": not intent.requires_subquery or signature["subqueries"] > 0,
        "set_operation": intent.set_operation == "none" or signature["set_operation"] == intent.set_operation,
        "result_shape": not result_data or not expected_grain_scalar or result_data["row_count"] <= 1,
        "result_limit_shape": not result_data or intent.limit is None or result_data["row_count"] <= intent.limit,
    }
    confidence = {
        "known_tables": "high", "projection_count": "high", "literal_values": "high",
        "aggregate_functions": "high", "having": "high", "distinct": "high", "ordering": "high",
        "ordering_direction": "high", "limit": "high", "unnecessary_limit": "high", "subquery": "medium",
        "set_operation": "high", "result_shape": "high", "result_limit_shape": "high",
        "required_tables": "high" if grounded.high_confidence else "low",
        "projection_bindings": "high" if required_projection else "low",
        "filter_bindings": "high" if required_filters else "low",
        "grouping_grain": "medium", "global_scalar_grain": "medium",
    }
    calibrated_rules = calibrated_rules or {name for name, level in confidence.items() if level == "high"}
    actionable = [name for name, passed in checks.items() if not passed and name in calibrated_rules]
    telemetry = [name for name, passed in checks.items() if not passed and name not in calibrated_rules]
    return {"version": CRITIC_VERSION, "checks": checks, "confidence": confidence,
            "actionable_diagnostics": actionable, "telemetry_diagnostics": telemetry,
            "passed": not actionable, "risk": len(actionable) / max(1, len(calibrated_rules)),
            "signature": signature, "result_signals": result_data,
            "source": "answer-free grounded intent, schema, value evidence, and safe execution signals"}


def calibrate_rules(reports: list[dict], correct: list[bool], *, minimum_precision: float = .90) -> dict:
    if len(reports) != len(correct) or not reports:
        raise ValueError("Critic calibration requires aligned nonempty reports and labels")
    names = sorted({name for report in reports for name in report.get("checks", {})})
    rows, allowed = {}, []
    for name in names:
        flagged = [index for index, report in enumerate(reports) if report.get("checks", {}).get(name) is False]
        true_flags = sum(not correct[index] for index in flagged)
        precision = true_flags / len(flagged) if flagged else None
        rows[name] = {"flagged": len(flagged), "true_flags": true_flags, "precision": precision}
        if precision is not None and precision >= minimum_precision:
            allowed.append(name)
    return {"version": "intent-critic-calibration-v1", "minimum_precision": minimum_precision,
            "rules": rows, "allowed_rules": allowed}


def select_candidate(candidates: list[dict]) -> tuple[dict, dict]:
    executed = [candidate for candidate in candidates if candidate.get("status") == "executed" and candidate.get("result")]
    if not executed:
        raise ValueError("No safe executed candidate to select")
    groups = Counter(candidate["result_signals"]["unordered_hash"] for candidate in executed)
    majority_hash, majority_count = groups.most_common(1)[0]
    majority = [candidate for candidate in executed if candidate["result_signals"]["unordered_hash"] == majority_hash]
    pool = majority if majority_count > 1 else executed
    ranked = sorted(pool, key=lambda candidate: (
        len(candidate.get("critic", {}).get("actionable_diagnostics", [])),
        float(candidate.get("critic", {}).get("risk", 1.0)),
        len(candidate.get("sql", "")),
        candidate.get("path", ""),
    ))
    selected = ranked[0]
    uncertain = majority_count == 1 and len({candidate["result_signals"]["unordered_hash"] for candidate in executed}) > 1
    reason = "majority_result_then_intent_critic" if majority_count > 1 else "intent_critic_no_result_majority"
    return selected, {"version": "sql-candidate-selection-v1", "reason": reason,
                      "result_groups": dict(groups), "selected_group": selected["result_signals"]["unordered_hash"],
                      "needs_review": uncertain or bool(selected.get("critic", {}).get("actionable_diagnostics"))}


def localized_repair(sql: str, component: str, grounded: GroundedIntent, evidence_facts: list[dict]) -> dict | None:
    if component not in {"literal_values", "distinct", "ordering_direction", "limit", "unnecessary_limit"}:
        return None
    tree = sqlglot.parse_one(sql, read="sqlite")
    before = structural_signature(sql)
    repaired = tree.copy()
    if component == "distinct" and isinstance(repaired, exp.Select):
        repaired.set("distinct", exp.Distinct())
    elif component == "limit" and isinstance(repaired, exp.Select) and grounded.semantic_intent.limit:
        repaired.set("limit", exp.Limit(expression=exp.Literal.number(grounded.semantic_intent.limit)))
    elif component == "unnecessary_limit" and isinstance(repaired, exp.Select):
        repaired.set("limit", None)
    elif component == "ordering_direction":
        order = next(repaired.find_all(exp.Order), None)
        direction = grounded.semantic_intent.ordering.direction if grounded.semantic_intent.ordering else "unknown"
        if not order or direction == "unknown":
            return None
        for ordered in order.expressions:
            ordered.set("desc", direction == "desc")
    elif component == "literal_values":
        expected = [str(row.value) for row in [*grounded.semantic_intent.filters, *grounded.semantic_intent.having]
                    if row.value is not None]
        supported = [str(row.get("printed_value")) for row in evidence_facts if row.get("kind") == "value"]
        replacements = [value for value in expected if value in supported]
        literals = list(repaired.find_all(exp.Literal))
        if len(replacements) != 1 or len(literals) != 1:
            return None
        literals[0].replace(exp.Literal.string(replacements[0]))
    repaired_sql = repaired.sql(dialect="sqlite")
    after = structural_signature(repaired_sql)
    allowed_dimensions = {
        "distinct": {"distinct"}, "limit": {"limit"}, "unnecessary_limit": {"limit"},
        "ordering_direction": {"ordering"}, "literal_values": {"predicates"},
    }[component]
    changed = {name for name in before if before[name] != after[name]}
    if not changed or not changed <= allowed_dimensions:
        return None
    return {"version": REPAIR_VERSION, "component": component, "before_sql": sql,
            "sql": repaired_sql, "changed_dimensions": sorted(changed), "broad_rewrite_rejected": False}
