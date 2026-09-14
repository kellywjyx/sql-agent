"""High-precision V10 detector, retrieval, and localized AST correction."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp

from .value_index import normalize_value
from .v10_memory import load as load_memory
from .v10_signature import ErrorSignature, build_signature, similarity


DETECTOR_VERSION = "sql-v10-error-detector-v1"
PATCH_VERSION = "sql-v10-local-patch-v1"


class Detection(BaseModel):
    version: str = DETECTOR_VERSION
    error_type: str | None = None
    error_subtype: str | None = None
    confidence: float = Field(ge=0, le=1)
    target: str | None = None
    patch: str | None = None
    rationale: str
    abstained: bool = True


def _select(tree):
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def _aggregates(tree):
    return list(tree.find_all(exp.AggFunc))


def _tables(tree) -> set[str]:
    return {node.name.casefold() for node in tree.find_all(exp.Table)}


def detect(question: str, sql: str, facts: list[dict], result: dict | None = None) -> Detection:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError as error:
        return Detection(confidence=0, rationale=f"unparseable SQL: {type(error).__name__}")
    signature = build_signature(question, facts, sql, result)
    flags, shape = signature.question_flags, signature.sql_shape
    aggregates = _aggregates(tree)
    requested = "count" if flags["asks_count"] else "avg" if flags["asks_average"] else "sum" if flags["asks_sum"] else None
    if requested and aggregates and aggregates[0].key != requested:
        return Detection(error_type="operation", error_subtype="wrong_aggregate", confidence=.96,
                         target="select_expression_0", patch=f"replace_aggregate:{requested}",
                         rationale=f"question explicitly requests {requested}; candidate uses {aggregates[0].key}", abstained=False)
    if (not requested and flags["asks_list"] and aggregates and not shape["group_by"] and
            shape["projection_count"] == 1 and aggregates[0].this is not None and
            not isinstance(aggregates[0].this, exp.Star)):
        return Detection(error_type="operation", error_subtype="unrequested_aggregate", confidence=.95,
                         target="select_expression_0", patch="remove_aggregate",
                         rationale="list/entity request produces an ungrouped scalar aggregate", abstained=False)
    if flags["asks_each"] and aggregates and not shape["group_by"] and shape["non_aggregate_projections"]:
        return Detection(error_type="grain", error_subtype="missing_group", confidence=.93,
                         target="group_by", patch="group_by_nonaggregate_projections",
                         rationale="per-entity aggregate has a non-aggregate projection but no GROUP BY", abstained=False)

    def safe_predicate_fact(fact: dict) -> bool:
        value = str(fact.get("db_value", ""))
        # Numeric-only literals commonly exist in several ID and measure
        # columns. V10 abstains unless the mapping is description-backed or an
        # alphanumeric identifier code, the only raw-value pattern that passed
        # calibration without a regression.
        mapped = fact.get("type") in {"encoded_value", "normalization"}
        identifier_code = bool(re.search(r"[A-Za-z]", value) and re.search(r"\d", value))
        return bool(mapped or identifier_code)

    value_facts = [fact for fact in facts if fact.get("db_value") is not None and fact.get("column") and
                   fact.get("type") in {"literal_value", "encoded_value", "normalization"} and
                   safe_predicate_fact(fact) and
                   normalize_value(str(fact.get("phrase", ""))) in normalize_value(question)]
    comparisons = [node for node in tree.walk() if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE))]
    for node in comparisons:
        column = node.this if isinstance(node.this, exp.Column) else node.expression if isinstance(node.expression, exp.Column) else None
        literal = node.expression if isinstance(node.expression, exp.Literal) else node.this if isinstance(node.this, exp.Literal) else None
        if not column or not literal:
            continue
        matches = [fact for fact in value_facts if normalize_value(str(fact["db_value"])) == normalize_value(str(literal.this))]
        if len(matches) == 1 and matches[0]["column"].casefold() != column.name.casefold() and \
                matches[0]["table"].casefold() in _tables(tree):
            fact = matches[0]
            return Detection(error_type="predicate", error_subtype="wrong_filter_column", confidence=.97,
                             target=column.sql(), patch=f"predicate_column:{fact['table']}:{fact['column']}",
                             rationale="one scoped FILTER_VALUE maps the used literal to a different in-query column",
                             abstained=False)

    join_facts = [fact for fact in facts if fact.get("type") == "join_relationship" and fact.get("column") and
                  fact.get("support", {}).get("target_table") and fact.get("support", {}).get("target_column")]
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        joined = join.this.name.casefold() if isinstance(join.this, exp.Table) else ""
        candidates = [fact for fact in join_facts if {fact["table"].casefold(),
                      fact["support"]["target_table"].casefold()} <= _tables(tree) and
                      joined in {fact["table"].casefold(), fact["support"]["target_table"].casefold()}]
        if len(candidates) == 1:
            fact = candidates[0]
            expected_columns = {fact["column"].casefold(), fact["support"]["target_column"].casefold()}
            actual_columns = {column.name.casefold() for column in on.find_all(exp.Column)} if on else set()
            if expected_columns != actual_columns:
                return Detection(error_type="join", error_subtype="wrong_join_predicate", confidence=.95,
                                 target="join_on", patch=(f"join:{fact['table']}:{fact['column']}:"
                                                          f"{fact['support']['target_table']}:"
                                                          f"{fact['support']['target_column']}"),
                                 rationale="candidate join conflicts with the only scoped relationship for its tables",
                                 abstained=False)
    return Detection(confidence=0, rationale="no high-confidence known-family error", abstained=True)


def retrieve(memory_path: Path, signature: ErrorSignature, family: str, *, limit: int = 3) -> list[dict]:
    rows = [row for row in load_memory(memory_path) if row["error_type"] == family]
    ranked = sorted(rows, key=lambda row: (-similarity(signature, ErrorSignature.model_validate(row["signature"])),
                                           row["identity"]))
    return [{"identity": row["identity"], "error_type": row["error_type"],
             "minimal_patch": row["minimal_patch"],
             "similarity": similarity(signature, ErrorSignature.model_validate(row["signature"])),
             "signature": row["signature"]} for row in ranked[:limit]]


def _alias_for(tree, table_name: str) -> str:
    for table in tree.find_all(exp.Table):
        if table.name.casefold() == table_name.casefold():
            return table.alias_or_name
    return table_name


def _fingerprint(tree, omitted: str) -> dict:
    select = _select(tree)
    values = {"from": select.args.get("from_").sql() if select and select.args.get("from_") else None,
              "joins": [row.sql() for row in select.args.get("joins", [])] if select else [],
              "where": select.args.get("where").sql() if select and select.args.get("where") else None,
              "group": select.args.get("group").sql() if select and select.args.get("group") else None,
              "order": select.args.get("order").sql() if select and select.args.get("order") else None,
              "limit": select.args.get("limit").sql() if select and select.args.get("limit") else None,
              "projections": [row.sql() for row in select.expressions] if select else []}
    values.pop(omitted, None)
    return values


def localized_patch(sql: str, detection: Detection) -> dict | None:
    if detection.abstained or not detection.patch:
        return None
    tree = sqlglot.parse_one(sql, read="sqlite"); before = tree.copy()
    component = ""
    if detection.patch == "remove_aggregate":
        node = next(tree.find_all(exp.AggFunc), None)
        if not node or node.this is None or isinstance(node.this, exp.Star): return None
        node.replace(node.this.copy()); component = "projections"
    elif detection.patch.startswith("replace_aggregate:"):
        node = next(tree.find_all(exp.AggFunc), None)
        if not node or node.this is None: return None
        name = detection.patch.split(":", 1)[1]
        factory = {"count": exp.Count, "sum": exp.Sum, "avg": exp.Avg}[name]
        node.replace(factory(this=node.this.copy())); component = "projections"
    elif detection.patch == "group_by_nonaggregate_projections":
        select = _select(tree)
        values = [row.copy() for row in select.expressions
                  if not next(row.find_all(exp.AggFunc), None)] if select else []
        if not values: return None
        select.set("group", exp.Group(expressions=values)); component = "group"
    elif detection.patch.startswith("predicate_column:"):
        _, table, column = detection.patch.split(":", 2)
        target_sql = detection.target
        target = next((row for row in tree.find_all(exp.Column) if row.sql() == target_sql), None)
        if not target: return None
        target.replace(exp.column(column, table=_alias_for(tree, table))); component = "where"
    elif detection.patch.startswith("join:"):
        _, left_table, left_column, right_table, right_column = detection.patch.split(":", 4)
        joins = list(tree.find_all(exp.Join))
        if len(joins) != 1: return None
        left_alias, right_alias = _alias_for(tree, left_table), _alias_for(tree, right_table)
        joins[0].set("on", exp.EQ(this=exp.column(left_column, table=left_alias),
                                  expression=exp.column(right_column, table=right_alias)))
        component = "joins"
    else:
        return None
    if _fingerprint(before, component) != _fingerprint(tree, component):
        return None
    patched = tree.sql(dialect="sqlite")
    return {"version": PATCH_VERSION, "sql": patched, "component": component,
            "before_component": _fingerprint(before, "").get(component),
            "after_component": _fingerprint(tree, "").get(component)}
