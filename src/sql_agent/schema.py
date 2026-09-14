from __future__ import annotations

import re
import sqlite3
from collections import deque
from pathlib import Path


def inspect_schema(database: Path) -> list[dict]:
    with sqlite3.connect(database.resolve(strict=True).as_uri() + "?mode=ro", uri=True) as connection:
        tables = connection.execute("SELECT name, sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        schema = []
        for name, ddl in tables:
            if "VIRTUAL TABLE" in (ddl or "").upper():
                raise ValueError("Virtual tables are unsupported in demonstration databases")
            escaped = name.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            foreign_keys = connection.execute(f'PRAGMA foreign_key_list("{escaped}")').fetchall()
            schema.append({"table": name, "ddl": ddl, "columns": [c[1] for c in columns],
                           "column_details": [{"name": c[1], "type": c[2], "not_null": bool(c[3]), "primary_key": bool(c[5])} for c in columns],
                           "foreign_keys": [{"table": f[2], "from": f[3], "to": f[4]} for f in foreign_keys],
                           "references": [f[2] for f in foreign_keys]})
        return schema


def linked_schema(question: str, schema: list[dict], enabled: bool = True) -> str:
    if not enabled or len(schema) <= 3:
        selected = schema
    else:
        terms = {x.rstrip("s") for x in re.findall(r"\w+", question.lower())}
        def score(table):
            words = {x.rstrip("s") for x in re.findall(r"\w+", (table["table"] + " " + " ".join(table["columns"])).lower().replace("_", " "))}
            return len(terms & words)
        ranked = sorted(schema, key=score, reverse=True)
        names = {t["table"] for t in ranked[:3]}
        graph = {t["table"]: set(t["references"]) for t in schema}
        for table in schema:
            for reference in table["references"]:
                graph.setdefault(reference, set()).add(table["table"])
        # Keep shortest join paths between seeds, independent of schema iteration order.
        seeds = sorted(names)
        for start in seeds:
            queue = deque([(start, [start])])
            visited = {start}
            while queue:
                node, path = queue.popleft()
                if node in seeds:
                    names.update(path)
                for neighbor in sorted(graph.get(node, [])):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))
        selected = [t for t in schema if t["table"] in names]
    return "\n\n".join(t["ddl"] for t in selected)


def value_hints(database: Path, question: str, schema: list[dict], max_probes: int = 4,
                selected_columns: list[dict] | None = None) -> list[dict]:
    """Inspect at most four text columns under the same read-only worker guards."""
    from .execution import execute_sql
    terms = re.findall(r"['\"]([^'\"\n]{1,60})['\"]", question)
    evidence = question.split("Benchmark evidence:", 1)[1] if "Benchmark evidence:" in question else ""
    terms += re.findall(r"(?:=|refers to)\s*['\"]?([A-Za-z][^;'\"\n]{1,40})", evidence, flags=re.IGNORECASE)
    terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))[:2]
    if not terms:
        return []
    hints, probes = [], 0
    def quote(identifier):
        return '"' + identifier.replace('"', '""') + '"'
    query_terms = set(re.findall(r"\w+", question.lower().replace("_", " ")))
    allowed = {(row["table"], row["column"]) for row in (selected_columns or [])}
    ranked = sorted(((table, column) for table in schema for column in table.get("column_details", [])
                     if not allowed or (table["table"], column["name"]) in allowed),
                    key=lambda pair: len(query_terms & set(re.findall(r"\w+", (pair[0]["table"] + " " + pair[1]["name"]).lower().replace("_", " ")))),
                    reverse=True)
    for table, column in ranked:
        if probes >= max_probes:
            return hints
        if not any(t in column["type"].upper() for t in ["TEXT", "CHAR", "CLOB"]):
            continue
        probes += 1
        literal = terms[(probes - 1) % len(terms)].replace("'", "''").replace("%", "\\%").replace("_", "\\_")
        sql = f"SELECT DISTINCT {quote(column['name'])} FROM {quote(table['table'])} WHERE {quote(column['name'])} LIKE '%{literal}%' ESCAPE '\\' LIMIT 3"
        try:
            result = execute_sql(database, sql, timeout=1, max_rows=3)
            if result["rows"]:
                hints.append({"table": table["table"], "column": column["name"],
                              "values": [str(row[0])[:120] for row in result["rows"]]})
        except (ValueError, TimeoutError, RuntimeError):
            continue
    return hints


def prompt_schema(question: str, schema: list[dict], byte_budget=10000) -> str:
    """Use full schema when it fits; otherwise rank columns, retaining join keys."""
    full = linked_schema(question, schema, False)
    if len(full.encode("utf-8")) <= byte_budget:
        return full
    terms = set(re.findall(r"\w+", question.lower().replace("_", " ")))
    def relevance(name):
        return len(terms & set(re.findall(r"\w+", name.lower().replace("_", " "))))
    by_name = {t["table"]: t for t in schema}
    names = {t["table"] for t in sorted(schema, key=lambda t: relevance(t["table"] + " " + " ".join(t["columns"])), reverse=True)[:3]}
    graph = {t["table"]: set(t["references"]) for t in schema}
    for table in schema:
        for other in table["references"]:
            graph.setdefault(other, set()).add(table["table"])
    seeds = sorted(names)
    for start in seeds:
        queue, visited = deque([(start, [start])]), {start}
        while queue:
            node, path = queue.popleft()
            if node in seeds:
                names.update(path)
            for other in sorted(graph.get(node, ())):
                if other not in visited:
                    visited.add(other)
                    queue.append((other, [*path, other]))
    selected = [by_name[name] for name in sorted(names) if name in by_name]
    required = {name: set() for name in names}
    for table in selected:
        for column in table.get("column_details", []):
            if column["primary_key"]:
                required[table["table"]].add(column["name"])
        for key in table.get("foreign_keys", []):
            if key["table"] in names:
                required[table["table"]].add(key["from"])
                required[key["table"]].add(key["to"])
    def quote(value):
        return '"' + value.replace('"', '""') + '"'
    def render(width):
        statements = []
        for table in selected:
            ranked = sorted(table["column_details"], key=lambda c: relevance(c["name"]), reverse=True)
            keep = required[table["table"]] | {c["name"] for c in ranked[:width]}
            columns = [quote(c["name"]) + " " + c["type"] for c in ranked if c["name"] in keep]
            columns += [f'FOREIGN KEY ({quote(k["from"])}) REFERENCES {quote(k["table"])} ({quote(k["to"])})'
                        for k in table.get("foreign_keys", []) if k["table"] in names and k["to"]]
            statements.append(f'CREATE TABLE {quote(table["table"])} (' + ", ".join(columns) + ");")
        return "\n".join(statements)
    for width in [32, 20, 12, 6, 0]:
        result = render(width)
        if len(result.encode("utf-8")) <= byte_budget:
            return result
    raise ValueError("Required join keys exceed the schema prompt budget")
