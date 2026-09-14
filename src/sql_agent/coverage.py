"""Question-to-query coverage diagnostics; never reads benchmark answers."""
from __future__ import annotations

import re
import sqlglot
from sqlglot import exp


def coverage_report(question: str, sql: str, schema: list[dict], plan: dict | None = None) -> dict:
    tree = sqlglot.parse_one(sql, read="sqlite")
    lowered = question.split("Benchmark evidence:", 1)[0].casefold()
    columns = {column["name"].casefold(): (table["table"], column["name"])
               for table in schema for column in table.get("column_details", [])}
    mentioned = sorted(name for name in columns if re.search(rf"(?<!\w){re.escape(name.replace('_', ' '))}(?!\w)", lowered.replace("_", " ")))
    sql_columns = {column.name.casefold() for column in tree.find_all(exp.Column)}
    aliases = {alias.alias.casefold() for alias in tree.find_all(exp.Alias) if alias.alias}
    sql_tables = {table.name.casefold() for table in tree.find_all(exp.Table)}
    known_tables = {table["table"].casefold() for table in schema}
    known_columns = set(columns)
    aggregate_kind = (plan or {}).get("aggregation")
    aggregate_required = bool(aggregate_kind) or any(term in lowered for term in ["how many", "count", "total", "sum", "average", "mean", "maximum", "minimum"])
    group_required = bool(plan and plan.get("grouping")) or aggregate_required and any(term in lowered for term in [" each ", " per ", " by ", "for every", "for each"])
    order_required = bool(plan and plan.get("ordering")) or any(term in lowered for term in ["highest", "lowest", "top ", "bottom ", "most ", "least ", "order", "sorted", "rank"])
    filter_required = bool((plan or {}).get("filter_columns")) or bool(re.search(
        r"['\"][^'\"]+['\"]|\bbetween\b|\bbefore\b|\bafter\b|\bduring\b|\b(?:in|for)\s+(?:19|20)\d{2}\b", lowered))
    join_required = bool(plan and plan.get("join_required")) or len({table for name in mentioned for table in [columns[name][0].casefold()]}) > 1
    required_outputs = {value.rsplit(".", 1)[-1].casefold() for value in (plan or {}).get("required_output_columns", [])}
    required_filters = {value.rsplit(".", 1)[-1].casefold() for value in (plan or {}).get("filter_columns", [])}
    where_columns = {column.name.casefold() for where in tree.find_all(exp.Where) for column in where.find_all(exp.Column)}
    group_columns = {column.name.casefold() for group in tree.find_all(exp.Group) for column in group.find_all(exp.Column)}
    projection_columns = {column.name.casefold() for select in tree.find_all(exp.Select)
                          for expression in select.expressions for column in expression.find_all(exp.Column)}
    graph = {table["table"].casefold(): set(name.casefold() for name in table.get("references", [])) for table in schema}
    for table in schema:
        for other in table.get("references", []):
            graph.setdefault(other.casefold(), set()).add(table["table"].casefold())
    connected = not sql_tables or len(sql_tables) == 1
    if len(sql_tables) > 1:
        start = next(iter(sql_tables)); seen = {start}; pending = [start]
        while pending:
            node = pending.pop()
            for other in graph.get(node, set()) & sql_tables:
                if other not in seen:
                    seen.add(other); pending.append(other)
        connected = sql_tables <= seen
    checks = {
        "known_tables": sql_tables <= known_tables,
        "known_columns": sql_columns <= known_columns | aliases | {"*"},
        "mentioned_columns": not mentioned or set(mentioned) <= sql_columns,
        "requested_projection": not required_outputs or bool(required_outputs & projection_columns),
        "requested_filters": not required_filters or bool(required_filters & where_columns),
        "aggregation": not aggregate_required or any(isinstance(node, exp.AggFunc) for node in tree.walk())
                       or (aggregate_kind in {"max", "min"} and tree.args.get("order") is not None and tree.args.get("limit") is not None),
        "grouping": not group_required or tree.args.get("group") is not None,
        "filtering": not filter_required or tree.args.get("where") is not None or any(select.args.get("where") is not None for select in tree.find_all(exp.Select)),
        "ordering": not order_required or tree.args.get("order") is not None,
        "limit": not (plan or {}).get("limit") or tree.args.get("limit") is not None,
        "joins": not join_required or any(True for _ in tree.find_all(exp.Join)),
        "foreign_key_connectivity": not join_required or connected,
    }
    diagnostics = [name for name, passed in checks.items() if not passed]
    return {"passed": not diagnostics, "checks": checks, "diagnostics": diagnostics,
            "question_requirements": {"columns": mentioned, "aggregation": aggregate_required,
                                      "grouping": group_required, "filtering": filter_required,
                                      "ordering": order_required, "joins": join_required,
                                      "limit": bool((plan or {}).get("limit")),
                                      "required_output_columns": sorted(required_outputs),
                                      "required_filter_columns": sorted(required_filters)},
            "query_references": {"tables": sorted(sql_tables), "columns": sorted(sql_columns),
                                 "projection_columns": sorted(projection_columns),
                                 "where_columns": sorted(where_columns), "group_columns": sorted(group_columns)}}
