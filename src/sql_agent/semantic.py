"""Answer-free SQL semantic checks and deterministic candidate ranking."""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from .coverage import coverage_report


SEMANTIC_VERSION = "sql-semantic-v1"


def _aliases(tree) -> dict[str, str]:
    result = {}
    for table in tree.find_all(exp.Table):
        # SQLGlot also represents table-valued expressions as Table nodes. A
        # physical identifier is therefore not guaranteed to be present.
        name = table.name
        if not isinstance(name, str) or not name:
            continue
        alias = table.alias_or_name or name
        result[str(alias).casefold()] = name.casefold()
        result[name.casefold()] = name.casefold()
    return result


def _column_pair(node, aliases: dict[str, str]):
    if not isinstance(node, exp.EQ) or not isinstance(node.left, exp.Column) or not isinstance(node.right, exp.Column):
        return None
    left = (aliases.get((node.left.table or "").casefold(), (node.left.table or "").casefold()), node.left.name.casefold())
    right = (aliases.get((node.right.table or "").casefold(), (node.right.table or "").casefold()), node.right.name.casefold())
    return left, right


def semantic_report(question: str, sql: str, schema: list[dict], plan: dict,
                    relevant_values: list[dict] | None = None) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    base = coverage_report(question, sql, schema, plan)
    aliases = _aliases(tree)
    foreign_pairs = set()
    foreign_table_pairs = set()
    primary_keys = {
        str(table.get("table", "")).casefold(): next(
            (str(column.get("name")) for column in table.get("column_details", [])
             if column.get("primary_key") and column.get("name")), None)
        for table in schema
    }
    for table in schema:
        source_name = table.get("table")
        if not isinstance(source_name, str) or not source_name:
            continue
        source = source_name.casefold()
        for key in table.get("foreign_keys", []):
            target_name = key.get("table")
            source_column = key.get("from")
            if not isinstance(target_name, str) or not target_name or not isinstance(source_column, str) or not source_column:
                continue
            target = target_name.casefold()
            # SQLite permits REFERENCES table without an explicit target
            # column. Resolve that form to the referenced primary key when it
            # is unambiguous instead of crashing or inventing a join.
            target_column = key.get("to") or primary_keys.get(target)
            if not isinstance(target_column, str) or not target_column:
                continue
            pair = ((source, source_column.casefold()), (target, target_column.casefold()))
            foreign_pairs.update([pair, (pair[1], pair[0])])
            foreign_table_pairs.add(frozenset([source, target]))
    invalid_join = False
    checked_join = False
    for join in tree.find_all(exp.Join):
        for equality in (join.args.get("on") or exp.Boolean(this=True)).find_all(exp.EQ):
            pair = _column_pair(equality, aliases)
            if not pair or not pair[0][0] or not pair[1][0]:
                continue
            tables = frozenset([pair[0][0], pair[1][0]])
            if tables in foreign_table_pairs:
                checked_join = True
                invalid_join |= pair not in foreign_pairs

    literals = {str(node.this).casefold() for node in tree.find_all(exp.Literal)}
    expected_literals = [str(value) for value in plan.get("expected_literals", [])]
    literal_matches = {value: value.casefold() in literals for value in expected_literals}
    expressions = list(tree.expressions) if isinstance(tree, exp.Select) else []
    expected_count = plan.get("expected_projection_count")
    aggregate_nodes = list(tree.find_all(exp.AggFunc))
    has_arithmetic = any(isinstance(node, (exp.Div, exp.Mul, exp.Sub, exp.Add)) for node in tree.walk())
    has_cast = any(isinstance(node, exp.Cast) for node in tree.walk())
    has_cleanup = has_cast or any(isinstance(node, (exp.Replace, exp.Substring)) for node in tree.walk())
    required_cleanup = plan.get("requires_numeric_cleanup", [])
    order = next(tree.find_all(exp.Order), None)
    order_sql = order.sql(dialect="sqlite").casefold() if order else ""
    direction = plan.get("ordering_direction")
    comparisons = {symbol for symbol, kind in [("<", exp.LT), (">", exp.GT), ("<=", exp.LTE), (">=", exp.GTE),
                                                     ("=", exp.EQ), ("!=", exp.NEQ)]
                   if any(isinstance(node, kind) for node in tree.walk())}
    expected_comparisons = set(plan.get("comparison_operators", []))
    checks = {
        **base["checks"],
        "projection_count": expected_count is None or len(expressions) >= expected_count,
        "literal_values": not expected_literals or all(literal_matches.values()),
        "comparison_operators": not expected_comparisons or bool(expected_comparisons & comparisons),
        "join_predicates": not checked_join or not invalid_join,
        "distinct": not plan.get("distinct") or bool(tree.args.get("distinct")),
        "ratio": not plan.get("ratio") or has_arithmetic,
        "percentage": not plan.get("percentage") or has_arithmetic,
        "numeric_cleanup": not required_cleanup or has_cleanup,
        "ordering_direction": direction is None or direction in order_sql,
        "unnecessary_limit": bool(plan.get("limit")) or tree.args.get("limit") is None,
    }
    hard_names = {"known_tables", "known_columns", "requested_projection", "requested_filters", "aggregation",
                  "grouping", "filtering", "ordering", "limit", "joins", "foreign_key_connectivity",
                  "literal_values", "comparison_operators", "join_predicates", "distinct", "ratio",
                  "percentage", "numeric_cleanup", "ordering_direction", "unnecessary_limit"}
    diagnostics = [name for name, passed in checks.items() if not passed]
    hard = [name for name in diagnostics if name in hard_names]
    risk = min(1.0, len(hard) / max(1, len(hard_names)))
    return {"version": SEMANTIC_VERSION, "passed": not hard, "checks": checks,
            "diagnostics": diagnostics, "hard_diagnostics": hard, "semantic_risk": risk,
            "details": {"literal_matches": literal_matches, "actual_comparisons": sorted(comparisons),
                        "aggregate_count": len(aggregate_nodes), "has_numeric_cleanup": has_cleanup,
                        "join_predicates_checked": checked_join},
            "query_references": base["query_references"], "question_requirements": base["question_requirements"]}


def enrich_plan_with_profiles(plan: dict, relevant_values: list[dict]) -> dict:
    enriched = dict(plan)
    numeric_tags = {"currency_text", "thousands_separator", "percentage_text", "plus_suffix", "numeric_text"}
    needs_numeric = bool(plan.get("aggregation") or plan.get("ordering") or plan.get("ratio"))
    enriched["requires_numeric_cleanup"] = sorted(
        f"{row['table']}.{row['column']}" for row in relevant_values
        if needs_numeric and numeric_tags & set(row.get("format_tags", [])))
    natural = " " + re.sub(r"\s+", " ", str(plan.get("natural_question", "")).casefold()) + " "
    if any(term in natural for term in [" highest ", " top ", " most ", " descending "]):
        enriched["ordering_direction"] = "desc"
    elif any(term in natural for term in [" lowest ", " bottom ", " least ", " ascending "]):
        enriched["ordering_direction"] = "asc"
    return enriched


def candidate_rank(candidate: dict) -> tuple:
    semantic = candidate.get("semantic_checks") or {}
    return (
        candidate.get("status") == "executed",
        semantic.get("passed", False),
        -len(semantic.get("hard_diagnostics", [])),
        -float(semantic.get("semantic_risk", 1.0)),
        -len(candidate.get("sql") or ""),
    )
