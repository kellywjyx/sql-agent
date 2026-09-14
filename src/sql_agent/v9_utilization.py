"""Scoring-only evidence utilization analysis for V9."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import sqlglot
from sqlglot import exp

from .value_index import normalize_value


METRIC_VERSION = "sql-v9-evidence-utilization-v1"


def _identifier(column: exp.Column, aliases: dict[str, str]) -> tuple[str, str]:
    table = aliases.get(column.table.casefold(), column.table).casefold() if column.table else ""
    return table, column.name.casefold()


def sql_effects(sql: str) -> list[dict]:
    tree = sqlglot.parse_one(sql, read="sqlite")
    aliases = {table.alias_or_name.casefold(): table.name for table in tree.find_all(exp.Table)}
    effects: list[dict] = []
    comparisons = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)
    for node in tree.walk():
        if isinstance(node, comparisons):
            column = node.this if isinstance(node.this, exp.Column) else node.expression if isinstance(node.expression, exp.Column) else None
            literal = node.expression if isinstance(node.expression, exp.Literal) else node.this if isinstance(node.this, exp.Literal) else None
            if column is not None:
                table, name = _identifier(column, aliases)
                effects.append({"kind": "predicate", "table": table, "column": name,
                                "operator": node.key, "value": normalize_value(literal.this) if literal else None})
        elif isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            table, name = _identifier(node.this, aliases)
            for literal in node.expressions:
                effects.append({"kind": "predicate", "table": table, "column": name,
                                "operator": "in", "value": normalize_value(literal.this) if isinstance(literal, exp.Literal) else None})
        elif isinstance(node, exp.Join) and node.args.get("on"):
            for equality in node.args["on"].find_all(exp.EQ):
                if isinstance(equality.this, exp.Column) and isinstance(equality.expression, exp.Column):
                    left, right = _identifier(equality.this, aliases), _identifier(equality.expression, aliases)
                    effects.append({"kind": "join", "columns": sorted([list(left), list(right)])})
        elif isinstance(node, exp.Cast):
            for column in node.find_all(exp.Column):
                table, name = _identifier(column, aliases)
                effects.append({"kind": "conversion", "table": table, "column": name,
                                "target": node.args["to"].sql().casefold() if node.args.get("to") else None})
    select = tree.find(exp.Select)
    if select:
        for expression in select.expressions:
            kind = "aggregation" if any(isinstance(node, exp.AggFunc) for node in expression.walk()) else "projection"
            for column in expression.find_all(exp.Column):
                table, name = _identifier(column, aliases)
                effects.append({"kind": kind, "table": table, "column": name})
    group = tree.args.get("group") or (select.args.get("group") if select else None)
    if group:
        for column in group.find_all(exp.Column):
            table, name = _identifier(column, aliases)
            effects.append({"kind": "grouping", "table": table, "column": name})
    order = tree.args.get("order") or (select.args.get("order") if select else None)
    if order:
        for column in order.find_all(exp.Column):
            table, name = _identifier(column, aliases)
            effects.append({"kind": "ordering", "table": table, "column": name})
    return effects


def _column_matches(fact: dict, effect: dict) -> bool:
    if effect.get("column") != str(fact.get("column") or "").casefold():
        return False
    table = str(fact.get("table") or "").casefold()
    return not table or not effect.get("table") or effect.get("table") == table


def _join_matches(fact: dict, effect: dict) -> bool:
    if effect.get("kind") != "join":
        return False
    source = [str(fact.get("table") or "").casefold(), str(fact.get("column") or "").casefold()]
    target = [str(fact.get("support", {}).get("target_table") or "").casefold(),
              str(fact.get("support", {}).get("target_column") or "").casefold()]
    return source in effect.get("columns", []) and target in effect.get("columns", [])


def expected_roles(fact: dict, reference_effects: list[dict]) -> list[str]:
    if fact.get("type") == "join_relationship":
        return ["join"] if any(_join_matches(fact, effect) for effect in reference_effects) else []
    matches = [effect for effect in reference_effects if _column_matches(fact, effect)]
    if fact.get("db_value") is not None:
        value = normalize_value(fact["db_value"])
        return ["predicate"] if any(effect["kind"] == "predicate" and effect.get("value") == value for effect in matches) else []
    if fact.get("type") == "numeric_or_storage":
        return sorted({effect["kind"] for effect in matches if effect["kind"] in {"conversion", "predicate", "aggregation"}})
    return sorted({effect["kind"] for effect in matches})


def observed_roles(fact: dict, prediction_effects: list[dict]) -> list[str]:
    if fact.get("type") == "join_relationship":
        return ["join"] if any(_join_matches(fact, effect) for effect in prediction_effects) else []
    matches = [effect for effect in prediction_effects if _column_matches(fact, effect)]
    if fact.get("db_value") is not None:
        value = normalize_value(fact["db_value"])
        exact = [effect for effect in matches if effect["kind"] == "predicate" and effect.get("value") == value]
        if exact:
            return ["predicate"]
    return sorted({effect["kind"] for effect in matches})


def classify_fact(fact: dict, reference_effects: list[dict], prediction_effects: list[dict]) -> dict:
    expected, observed = expected_roles(fact, reference_effects), observed_roles(fact, prediction_effects)
    if expected and set(expected) & set(observed):
        label = "correctly_used"
    elif expected and observed:
        label = "misapplied"
    elif expected:
        label = "ignored"
    elif observed:
        label = "unnecessarily_used"
    else:
        label = "ignored"
    if expected and "predicate" in expected:
        same_column_predicates = [effect for effect in prediction_effects
                                  if effect.get("kind") == "predicate" and _column_matches(fact, effect)]
        value = normalize_value(fact.get("db_value", ""))
        if same_column_predicates and not any(effect.get("value") == value for effect in same_column_predicates):
            label = "contradicted"
    return {"evidence": fact, "expected_roles": expected, "observed_roles": observed, "utilization": label}


def analyze(reference_sql: str, predicted_sql: str | None, facts: list[dict]) -> dict:
    reference = sql_effects(reference_sql)
    try:
        predicted = sql_effects(predicted_sql) if predicted_sql else []
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        predicted = []
    rows = [classify_fact(fact, reference, predicted) for fact in facts]
    counts = Counter(row["utilization"] for row in rows)
    relevant = sum(bool(row["expected_roles"]) for row in rows)
    correct = counts["correctly_used"]
    irrelevant = len(rows) - relevant
    return {"version": METRIC_VERSION, "facts": rows, "counts": dict(counts),
            "relevant_fact_count": relevant, "irrelevant_fact_count": irrelevant,
            "evidence_utilization_accuracy": correct / relevant if relevant else None,
            "unnecessary_use_rate": counts["unnecessarily_used"] / irrelevant if irrelevant else None}


def aggregate(reports: Iterable[dict]) -> dict:
    reports = list(reports)
    counts = Counter()
    relevant = correct = irrelevant = unnecessary = 0
    for report in reports:
        counts.update(report["counts"])
        relevant += report["relevant_fact_count"]
        irrelevant += report["irrelevant_fact_count"]
        correct += report["counts"].get("correctly_used", 0)
        unnecessary += report["counts"].get("unnecessarily_used", 0)
    return {"version": METRIC_VERSION, "cases": len(reports), "counts": dict(counts),
            "evidence_utilization_accuracy": correct / relevant if relevant else None,
            "unnecessary_use_rate": unnecessary / irrelevant if irrelevant else None,
            "relevant_fact_count": relevant, "irrelevant_fact_count": irrelevant}
