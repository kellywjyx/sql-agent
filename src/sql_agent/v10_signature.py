"""Schema-independent error signatures and four-family labels for V10."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp


SIGNATURE_VERSION = "sql-v10-error-signature-v1"
ErrorFamily = Literal["operation", "grain", "predicate", "join"]


class ErrorSignature(BaseModel):
    version: Literal["sql-v10-error-signature-v1"] = SIGNATURE_VERSION
    question_flags: dict[str, bool]
    evidence_roles: list[str] = Field(default_factory=list)
    sql_shape: dict
    result_shape: dict


def question_flags(question: str) -> dict[str, bool]:
    value = " " + question.casefold().replace("_", " ") + " "
    patterns = {
        "asks_count": r"\b(how many|number of|count)\b",
        "asks_average": r"\b(average|avg|mean)\b",
        "asks_sum": r"\b(total|sum|combined)\b",
        "asks_each": r"\b(each|every|per|for each|by each)\b",
        "asks_list": r"\b(list|show|give|find|which|what are|return)\b",
        "asks_top_k": r"\b(top|highest|lowest|most|least|largest|smallest)\b",
        "asks_distinct": r"\b(distinct|different|unique)\b",
    }
    return {name: bool(re.search(pattern, value)) for name, pattern in patterns.items()}


def evidence_roles(facts: list[dict]) -> list[str]:
    mapping = {"literal_value": "filter_value", "encoded_value": "filter_value",
               "normalization": "filter_value", "numeric_or_storage": "storage",
               "join_relationship": "join", "column_semantics": "identifier_binding"}
    return sorted({mapping.get(str(fact.get("type")), "other") for fact in facts})


def sql_shape(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projections = list(select.expressions) if select else []
    aggregates = [node.key for node in tree.find_all(exp.AggFunc)]
    predicates = [node.key for node in tree.walk()
                  if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In))]
    non_aggregate_projections = sum(not any(isinstance(node, exp.AggFunc) for node in expression.walk())
                                    for expression in projections)
    return {"aggregates": aggregates, "has_aggregate": bool(aggregates),
            "group_by": bool(select and select.args.get("group")),
            "distinct": bool(select and select.args.get("distinct")),
            "join_count": len(list(tree.find_all(exp.Join))), "projection_count": len(projections),
            "non_aggregate_projections": non_aggregate_projections,
            "predicate_count": len(predicates), "predicate_operators": predicates,
            "subquery": bool(next(tree.find_all(exp.Subquery), None))}


def build_signature(question: str, facts: list[dict], sql: str, result: dict | None = None) -> ErrorSignature:
    result = result or {}
    return ErrorSignature(question_flags=question_flags(question), evidence_roles=evidence_roles(facts),
                          sql_shape=sql_shape(sql), result_shape={"rows": int(result.get("row_count", len(result.get("rows", [])))),
                                                                 "columns": len(result.get("columns", []))})


def canonical_identity(signature: ErrorSignature, family: str, patch: str) -> str:
    payload = {"signature": signature.model_dump(), "family": family, "patch": patch}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def similarity(left: ErrorSignature, right: ErrorSignature) -> float:
    flags = set(name for name, value in left.question_flags.items() if value)
    other_flags = set(name for name, value in right.question_flags.items() if value)
    flag_score = len(flags & other_flags) / max(1, len(flags | other_flags))
    roles, other_roles = set(left.evidence_roles), set(right.evidence_roles)
    role_score = len(roles & other_roles) / max(1, len(roles | other_roles))
    fields = ["has_aggregate", "group_by", "join_count", "projection_count", "predicate_count"]
    shape_score = sum(left.sql_shape.get(field) == right.sql_shape.get(field) for field in fields) / len(fields)
    return .45 * flag_score + .20 * role_score + .35 * shape_score
