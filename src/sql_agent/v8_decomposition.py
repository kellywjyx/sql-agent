"""Scoring-only oracle-help and recoverability decomposition for V8."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlglot
from llm_evals.campaign import immutable_json
from sqlglot import exp

from .guardrails import authorizer
from .m_schema import split_question_evidence
from .schema import inspect_schema
from .schema_retrieval import schema_documents, tokens
from .v7_evaluation import V7Scoring
from .v8_data import open_cases


TAXONOMY_VERSION = "sql-v8-evidence-taxonomy-v1"
STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "for", "and", "or", "that",
        "this", "refers", "means", "value", "values", "return", "find", "give", "show"}


def _predictions(root: Path, mode: str, run_id: str) -> list[dict]:
    path = root / f"v7/sql-agent/eval/reproduction/{mode}/{run_id}/predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def evidence_types(question: str, evidence: str) -> list[str]:
    text = evidence.casefold()
    types = []
    if re.search(r"\b(percent|percentage|ratio|difference|increase|decrease|sum|average|subtract|divide|multiply)\b|[+*/]", text):
        types.append("numeric_formula")
    if re.search(r"\b(thousand|million|billion|unit|scale|decimal)\b", text):
        types.append("numeric_scaling_or_unit")
    if re.search(r"\b(fiscal|date|year|month|day|quarter|semester)\b", text):
        types.append("date_interpretation")
    if re.search(r"\b(cast|stored as|text|integer|real|numeric|string)\b", text):
        types.append("cast_or_storage_rule")
    if re.search(r"\b(join|relationship|related|references?)\b", text):
        types.append("join_meaning")
    if re.search(r"[`'\"]", evidence):
        types.append("exact_value")
    if re.search(r"\b(0|1|y|n|yes|no|true|false|code|status)\b.*\b(mean|indicat|refer|represent|denot)", text):
        types.append("encoded_category")
    if re.search(r"\b(synonym|same as|also called|refers to|means)\b", text):
        types.append("synonym_or_concept_mapping")
    if re.search(r"\b(column|field|attribute)\b", text) or re.search(r"`[^`]+`", evidence):
        types.append("column_semantics")
    natural = question.split("Benchmark evidence:", 1)[0].casefold()
    if re.search(r"\b(top|bottom|most|least|highest|lowest|rank|each|per|among)\b", natural):
        types.append("multi_step_logic")
    if not types and evidence.strip():
        types.append("domain_rule")
    return list(dict.fromkeys(types))


def _reference_targets(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    aliases = {table.alias_or_name.casefold(): table.name for table in tree.find_all(exp.Table)}
    tables = {table.name for table in tree.find_all(exp.Table)}
    columns = []
    for column in tree.find_all(exp.Column):
        table = aliases.get(column.table.casefold(), column.table) if column.table else None
        columns.append({"table": table, "column": column.name})
    literals = []
    comparison_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)
    for node in tree.walk():
        if isinstance(node, comparison_types):
            column = node.this if isinstance(node.this, exp.Column) else node.expression if isinstance(node.expression, exp.Column) else None
            literal = node.expression if isinstance(node.expression, exp.Literal) else node.this if isinstance(node.this, exp.Literal) else None
            if column is not None and literal is not None:
                literals.append({"table": aliases.get(column.table.casefold(), column.table) if column.table else None,
                                 "column": column.name, "value": str(literal.this),
                                 "is_string": bool(literal.is_string)})
        elif isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            for literal in node.expressions:
                if isinstance(literal, exp.Literal):
                    literals.append({"table": aliases.get(node.this.table.casefold(), node.this.table) if node.this.table else None,
                                     "column": node.this.name, "value": str(literal.this),
                                     "is_string": bool(literal.is_string)})
    transformations = []
    if any(isinstance(node, exp.Cast) for node in tree.walk()): transformations.append("cast")
    if any(isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)) for node in tree.walk()): transformations.append("numeric_formula")
    # SQLGlot's concrete date-expression classes vary between releases.
    if any(getattr(node, "key", "") in {"date", "datetrunc", "strtounix", "timetostr", "tsordstodate"}
           or str(getattr(node, "name", "")).casefold() in {"date", "strftime"}
           for node in tree.walk()):
        transformations.append("date")
    joins = [{"left": join.this.name if isinstance(join.this, exp.Table) else join.this.sql(),
              "on": join.args.get("on").sql() if join.args.get("on") else None}
             for join in tree.find_all(exp.Join)]
    return {"tables": sorted(tables), "columns": columns, "literals": literals,
            "transformations": transformations, "joins": joins}


def _value_exists(database: Path, schemas: list[dict], target: dict) -> bool:
    candidates = []
    for table in schemas:
        if target.get("table") and table["table"].casefold() != str(target["table"]).casefold():
            continue
        if any(column["name"].casefold() == target["column"].casefold() for column in table["column_details"]):
            candidates.append(table["table"])
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.enable_load_extension(False)
        connection.set_authorizer(authorizer)
        for table in candidates:
            q = lambda value: '"' + value.replace('"', '""') + '"'
            try:
                row = connection.execute(f"SELECT 1 FROM {q(table)} WHERE {q(target['column'])} = ? LIMIT 1",
                                         (target["value"],)).fetchone()
            except sqlite3.DatabaseError:
                continue
            if row:
                return True
    return False


def recoverability(case, evidence: str, types_found: list[str]) -> dict:
    database = Path(case.context["database"])
    schema = inspect_schema(database)
    documents = schema_documents(database, schema)
    targets = _reference_targets(case.expected["sql"])
    evidence_terms = {term for term in tokens(evidence) if term not in STOP and len(term) > 2}
    descriptions = " ".join(row.get("text", "") for row in documents)
    sources = []
    identifier_terms = {term for table in schema for term in tokens(table["table"] + " " + " ".join(table["columns"]))}
    if evidence_terms & identifier_terms:
        sources.append("schema_identifier")
    if len(evidence_terms & set(tokens(descriptions))) >= 2:
        sources.append("database_description")
    if any(_value_exists(database, schema, literal) for literal in targets["literals"]):
        sources.append("database_values")
    if targets["joins"] and any(table.get("foreign_keys") for table in schema):
        sources.append("foreign_keys")
    if targets["transformations"] or set(types_found) & {"numeric_scaling_or_unit", "cast_or_storage_rule"}:
        sources.append("value_distribution_or_statistics")
    if set(types_found) & {"numeric_formula", "multi_step_logic"}:
        sources.append("reasoning_derived")
    database_sources = [source for source in sources if source not in {"reasoning_derived"}]
    if not sources:
        sources = ["not_recoverable_without_external_knowledge"]
    elif len(database_sources) > 1:
        sources.append("combination")
    return {"sources": list(dict.fromkeys(sources)),
            "database_recoverable": bool(database_sources),
            "reference_targets": targets}


def build(root: Path) -> dict:
    root = root.resolve()
    target = root / "v8/sql-agent/decomposition.jsonl"
    report_path = root / "v8/sql-agent/decomposition-report.json"
    if target.exists() and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads((root / "v8/sql-agent/protocol.json").read_text(encoding="utf-8"))
    cases = open_cases(root)
    none = {row["id"]: row for row in _predictions(root, "reference_none", protocol["controls"]["no_evidence_run"])}
    oracle = {row["id"]: row for row in _predictions(root, "reference_oracle", protocol["controls"]["oracle_evidence_run"])}
    scorer = V7Scoring(cases, official_scorer_path=root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    rows, counts = [], {letter: 0 for letter in "ABCD"}
    taxonomy_assigned = recoverability_assigned = 0
    for case in cases:
        left, right = scorer.correct(case, none[case.id]), scorer.correct(case, oracle[case.id])
        group = "B" if left and right else "D" if left else "A" if right else "C"
        _, evidence = split_question_evidence(case.input)
        categories = evidence_types(case.input, evidence)
        recoverable = recoverability(case, evidence, categories)
        counts[group] += 1
        taxonomy_assigned += bool(categories)
        recoverability_assigned += bool(recoverable["sources"])
        rows.append({"id": case.id, "db_id": case.metadata["db_id"], "difficulty": case.metadata.get("difficulty"),
                     "group": group, "no_evidence_correct": left, "oracle_evidence_correct": right,
                     "evidence_types": categories, "recoverability": recoverable,
                     "oracle_evidence": evidence, "label_method": "deterministic_scoring_and_rule_taxonomy",
                     "needs_manual_review": not categories or recoverable["sources"] == ["not_recoverable_without_external_knowledge"]})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    group_a = [row for row in rows if row["group"] == "A"]
    report = {"version": TAXONOMY_VERSION, "count": len(rows), "groups": counts,
              "group_a_count": len(group_a),
              "group_a_database_recoverable": sum(row["recoverability"]["database_recoverable"] for row in group_a),
              "taxonomy_coverage": taxonomy_assigned / len(rows),
              "recoverability_coverage": recoverability_assigned / len(rows),
              "manual_review_count": sum(row["needs_manual_review"] for row in rows),
              "gate_a_passed": taxonomy_assigned / len(rows) >= .90 and recoverability_assigned / len(rows) >= .90,
              "contains_oracle_evidence": True, "scoring_only": True, "locked_final_opened": False}
    immutable_json(report_path, report)
    return report
