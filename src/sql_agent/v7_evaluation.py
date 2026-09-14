"""Scoring-only V7 metrics. None of these helpers belong in application prompts."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
from pathlib import Path

import sqlglot
from llm_evals import Metric
from sqlglot import exp

from .evaluation import equivalent
from .execution import execute_sql
from .semantic_ir import reference_structure
from .value_index import normalize_value


METRIC_VERSION = "sql-v7-metrics-v1"
OFFICIAL_SCORER_SHA256 = "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89"


def load_official_calculate_ex(path: Path, expected_sha256: str = OFFICIAL_SCORER_SHA256):
    """Load only Mini-Dev's checksum-pinned ``calculate_ex`` function.

    Importing the complete upstream module would execute imports for its CLI and
    multiprocessing evaluator.  Extracting this single function keeps the cross
    score faithful while avoiding unrelated, optional benchmark dependencies.
    """
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Official evaluator checksum mismatch: expected {expected_sha256}, got {actual}")
    module = ast.parse(source.decode("utf-8"), filename=str(path))
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef)
               and node.name == "calculate_ex"]
    if len(matches) != 1:
        raise ValueError("Pinned official evaluator must define exactly one calculate_ex function")
    isolated = ast.Module(body=matches, type_ignores=[])
    namespace: dict = {}
    exec(compile(isolated, str(path), "exec"), {"__builtins__": {"set": set, "int": int}}, namespace)
    return namespace["calculate_ex"]


class V7Scoring:
    def __init__(self, cases, official_scorer_path: Path | None = None):
        self.cases = cases
        self.official_calculate_ex = (load_official_calculate_ex(official_scorer_path)
                                      if official_scorer_path else None)
        self.gold, self.errors = {}, {}
        for case in cases:
            try:
                self.gold[case.id] = execute_sql(__import__("pathlib").Path(case.context["database"]),
                                                 case.expected["sql"], timeout=10)
            except (ValueError, RuntimeError, TimeoutError) as error:
                self.errors[case.id] = str(error)

    def correct(self, case, prediction) -> bool:
        return bool(case.id in self.gold and prediction.get("status") == "completed" and
                    prediction.get("rows") is not None and prediction.get("columns") is not None and
                    equivalent(self.gold[case.id], prediction, case.expected.get("ordered", False)))

    def candidate_correct(self, case, candidate) -> bool:
        result = candidate.get("result") or {}
        return bool(case.id in self.gold and candidate.get("status") == "executed" and
                    equivalent(self.gold[case.id], result, case.expected.get("ordered", False)))

    def execution_accuracy(self, cases, predictions):
        return sum(self.correct(case, prediction) for case, prediction in zip(cases, predictions)) / len(cases)

    def completion(self, cases, predictions):
        return sum(prediction.get("status") == "completed" for prediction in predictions) / len(cases)

    def first_candidate_accuracy(self, cases, predictions):
        correct = 0
        for case, prediction in zip(cases, predictions):
            candidates = prediction.get("candidates", [])
            if candidates and self.candidate_correct(case, candidates[0]):
                correct += 1
        return correct / len(cases)

    def candidate_oracle(self, cases, predictions):
        return sum(any(self.candidate_correct(case, candidate) for candidate in prediction.get("candidates", []))
                   for case, prediction in zip(cases, predictions)) / len(cases)

    def official_set_equivalent(self, gold: dict, predicted: dict) -> bool:
        gold_rows = list(map(tuple, gold.get("rows", [])))
        predicted_rows = list(map(tuple, predicted.get("rows", [])))
        if self.official_calculate_ex:
            return bool(self.official_calculate_ex(predicted_rows, gold_rows))
        return set(predicted_rows) == set(gold_rows)

    def official_gold_self_consistency(self, cases, predictions):
        available = [self.gold[case.id] for case in cases if case.id in self.gold]
        if not available:
            return 0
        return sum(self.official_set_equivalent(result, result) for result in available) / len(available)

    def scorer_agreement(self, cases, predictions):
        comparable = 0
        agreed = 0
        for case, prediction in zip(cases, predictions):
            if case.id not in self.gold or prediction.get("status") != "completed":
                continue
            local = equivalent(self.gold[case.id], prediction, case.expected.get("ordered", False))
            official = self.official_set_equivalent(self.gold[case.id], prediction)
            comparable += 1
            agreed += local == official
        return agreed / comparable if comparable else 0

    def evidence_metrics(self, cases, predictions) -> dict:
        table_scores, column_scores, literal_scores = [], [], []
        for case, prediction in zip(cases, predictions):
            try:
                tree = sqlglot.parse_one(case.expected["sql"], read="sqlite")
            except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
                continue
            required_tables = {node.name.casefold() for node in tree.find_all(exp.Table)}
            required_columns = {node.name.casefold() for node in tree.find_all(exp.Column)}
            required_literals = {normalize_value(node.this) for node in tree.find_all(exp.Literal)
                                 if isinstance(node.this, str) and len(str(node.this)) > 0}
            selection = prediction.get("schema_selection") or {}
            selected_tables = {str(value).casefold() for value in selection.get("selected_tables", [])}
            selected_columns = {str(row.get("column", "")).casefold() for row in selection.get("selected_columns", [])}
            packet_values = {normalize_value(row.get("printed_value", ""))
                             for row in prediction.get("evidence_packet", []) if row.get("kind") == "value"}
            if required_tables:
                table_scores.append(len(required_tables & selected_tables) / len(required_tables))
            if required_columns:
                column_scores.append(len(required_columns & selected_columns) / len(required_columns))
            if required_literals:
                literal_scores.append(len(required_literals & packet_values) / len(required_literals))
        average = lambda values: sum(values) / len(values) if values else 0
        return {"evidence_table_recall": average(table_scores), "evidence_column_recall": average(column_scores),
                "evidence_literal_recall": average(literal_scores), "scorable_literal_cases": len(literal_scores)}

    def ir_structural_accuracy(self, cases, predictions):
        values = []
        for case, prediction in zip(cases, predictions):
            intent = prediction.get("semantic_intent")
            if not intent:
                continue
            reference = reference_structure(case.expected["sql"])
            expected_aggregates = {str(row.get("aggregation")) for row in intent.get("measures", [])
                                   if row.get("aggregation") != "none"}
            components = [
                len(intent.get("projections", [])) + len(intent.get("measures", [])) == reference["projection_count"],
                bool(expected_aggregates) == bool(reference["aggregates"]),
                bool(intent.get("grouping")) == reference["grouping"],
                bool(intent.get("having")) == reference["having"],
                bool(intent.get("distinct_required")) == reference["distinct"],
                bool(intent.get("ordering")) == reference["ordering"],
                bool(intent.get("limit")) == reference["limit"],
                bool(intent.get("requires_subquery")) == reference["subquery"],
                str(intent.get("set_operation", "none")) == reference["set_operation"],
            ]
            values.extend(components)
        if not values:
            raise ValueError("not applicable: no prediction carries a semantic_intent")
        return sum(values) / len(values)

    def metrics(self, *, include_ir: bool = True):
        # ir_structural_accuracy applies only to pipelines that emit a semantic intent (V7 grounded).
        # Other pipelines opt out instead of reporting a misleading 0.0.
        return [
            Metric("execution_accuracy_v7", self.execution_accuracy, version=METRIC_VERSION),
            Metric("completion", self.completion, version=METRIC_VERSION),
            Metric("first_candidate_accuracy", self.first_candidate_accuracy, version=METRIC_VERSION),
            Metric("candidate_oracle_ex", self.candidate_oracle, version=METRIC_VERSION),
            Metric("scorer_agreement", self.scorer_agreement, version=METRIC_VERSION),
            Metric("official_gold_self_consistency", self.official_gold_self_consistency,
                   version=METRIC_VERSION),
        ] + ([Metric("ir_structural_accuracy", self.ir_structural_accuracy, version=METRIC_VERSION)] if include_ir else [])


def structural_error_families(expected_sql: str, actual_sql: str | None) -> list[str]:
    if not actual_sql:
        return ["application_failure"]
    try:
        expected, actual = reference_structure(expected_sql), reference_structure(actual_sql)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        return ["syntax"]
    names = {
        "tables": "table", "columns": "column", "projection_count": "projection",
        "aggregates": "aggregate", "grouping": "grain", "having": "having",
        "distinct": "distinct", "ordering": "ordering", "ordering_direction": "ordering_direction",
        "limit": "limit", "joins": "join", "subquery": "subquery", "set_operation": "set_operation",
    }
    return [label for field, label in names.items() if expected[field] != actual[field]] or ["result_semantics"]
