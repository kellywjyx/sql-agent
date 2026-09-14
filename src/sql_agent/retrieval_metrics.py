"""Gold-aware schema recall used only by evaluation scoring."""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from .schema import inspect_schema


def required_schema(case) -> tuple[set[str], set[str]]:
    tree = sqlglot.parse_one(case.expected["sql"], read="sqlite")
    tables = {table.name.casefold() for table in tree.find_all(exp.Table)}
    aliases = {alias.alias.casefold() for alias in tree.find_all(exp.Alias) if alias.alias}
    columns = {column.name.casefold() for column in tree.find_all(exp.Column)
               if column.name and column.name != "*" and column.name.casefold() not in aliases}
    return tables, columns


def retrieval_scores(case, prediction) -> dict:
    try:
        tables, columns = required_schema(case)
    except sqlglot.errors.SqlglotError:
        return {"reference_parsed": False, "table_hits": 0, "table_total": 0,
                "column_hits": 0, "column_total": 0, "path_covered": False, "path_required": False}
    schema = inspect_schema(__import__("pathlib").Path(case.context["database"]))
    actual_tables = {table["table"].casefold() for table in schema}
    actual_columns = {column.casefold() for table in schema for column in table["columns"]}
    invalid_tables, invalid_columns = tables - actual_tables, columns - actual_columns
    tables, columns = tables & actual_tables, columns & actual_columns
    selection = prediction.get("schema_selection") or {}
    selected_tables = {name.casefold() for name in selection.get("selected_tables", [])}
    selected_columns = {row["column"].casefold() for row in selection.get("selected_columns", [])}
    selected_columns.update(value.split(".", 1)[1].casefold() for value in selection.get("retained_keys", []) if "." in value)
    path_required = len(tables) > 1
    return {"reference_parsed": True, "table_hits": len(tables & selected_tables), "table_total": len(tables),
            "column_hits": len(columns & selected_columns), "column_total": len(columns),
            "path_covered": not path_required or tables <= selected_tables, "path_required": path_required,
            "invalid_table_references": len(invalid_tables), "invalid_column_references": len(invalid_columns)}


def aggregate_retrieval(cases, predictions) -> dict:
    rows = [retrieval_scores(case, prediction) for case, prediction in zip(cases, predictions)]
    table_total = sum(row["table_total"] for row in rows)
    column_total = sum(row["column_total"] for row in rows)
    paths = [row for row in rows if row["path_required"]]
    return {"schema_table_recall": sum(row["table_hits"] for row in rows) / table_total if table_total else 0,
            "schema_column_recall": sum(row["column_hits"] for row in rows) / column_total if column_total else 0,
            "foreign_key_path_coverage": sum(row["path_covered"] for row in paths) / len(paths) if paths else 1,
            "reference_parse_coverage": sum(row["reference_parsed"] for row in rows) / len(rows),
            "invalid_reference_tables": sum(row.get("invalid_table_references", 0) for row in rows),
            "invalid_reference_columns": sum(row.get("invalid_column_references", 0) for row in rows)}
