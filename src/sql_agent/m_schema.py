"""Compact M-Schema-style rendering for SQLite prompts."""
from __future__ import annotations

import csv
import re
from pathlib import Path


M_SCHEMA_VERSION = "m-schema-sqlite-v1"


def split_question_evidence(value: str) -> tuple[str, str]:
    marker = "Benchmark evidence:"
    if marker not in value:
        return value.strip(), ""
    question, evidence = value.split(marker, 1)
    return question.strip(), evidence.strip()


def _descriptions(database: Path) -> dict[tuple[str, str], str]:
    folder = database.parent / "database_description"
    rows: dict[tuple[str, str], str] = {}
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    name = row.get("original_column_name") or row.get("column_name")
                    description = row.get("column_description") or ""
                    if name:
                        rows[(path.stem.casefold(), name.casefold())] = description.strip()[:300]
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows


def render_m_schema(database: Path, question: str, schema: list[dict], *, profiles: list[dict] | None = None,
                    selected_columns: list[dict] | None = None, byte_budget: int = 10000) -> tuple[str, dict]:
    descriptions = _descriptions(database)
    profile_map = {(row.get("table"), row.get("column")): row for row in (profiles or [])}
    selected = {(row["table"], row["column"]) for row in (selected_columns or [])}
    terms = set(re.findall(r"[a-z0-9]+", question.casefold().replace("_", " ")))
    entries = []
    for table in schema:
        keys = {key["from"]: key for key in table.get("foreign_keys", [])}
        for column in table.get("column_details", []):
            table_name, name = table["table"], column["name"]
            profile = profile_map.get((table_name, name), {})
            roles = []
            if column.get("primary_key"):
                roles.append("PK")
            if name in keys:
                roles.append(f"FK->{keys[name]['table']}.{keys[name]['to']}")
            description = descriptions.get((table_name.casefold(), name.casefold()), "")
            extras = []
            if roles:
                extras.append("/".join(roles))
            if description:
                extras.append(description)
            if profile.get("format_tags"):
                extras.append("storage=" + ",".join(profile["format_tags"]))
            if profile.get("values"):
                extras.append("examples=" + repr(profile["values"][:3]))
            line = f"  - {name} {column.get('type') or 'UNKNOWN'}" + (" | " + " | ".join(extras) if extras else "")
            words = set(re.findall(r"[a-z0-9]+", f"{table_name} {name} {description}".casefold().replace("_", " ")))
            priority = (100 if (table_name, name) in selected else 0) + 20 * bool(roles) + 5 * len(terms & words)
            entries.append({"table": table_name, "column": name, "line": line, "priority": priority,
                            "required": bool(roles) or (table_name, name) in selected})

    def render(rows: list[dict]) -> str:
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["table"], []).append(row["line"])
        blocks = [f"[Database: SQLite; schema-format: {M_SCHEMA_VERSION}]" ]
        for table_name in sorted(grouped):
            neighbours = sorted({key["table"] for table in schema if table["table"] == table_name
                                 for key in table.get("foreign_keys", [])})
            suffix = f"; neighbours={','.join(neighbours)}" if neighbours else ""
            blocks.append(f"[Table] {table_name}{suffix}\n" + "\n".join(grouped[table_name]))
        return "\n\n".join(blocks)

    complete = render(entries)
    if len(complete.encode("utf-8")) <= byte_budget:
        kept = entries
    else:
        kept = [row for row in entries if row["required"]]
        for row in sorted(entries, key=lambda value: (-value["priority"], value["table"], value["column"])):
            if row in kept:
                continue
            candidate = [*kept, row]
            if len(render(candidate).encode("utf-8")) <= byte_budget:
                kept.append(row)
        complete = render(kept)
    if len(complete.encode("utf-8")) > byte_budget:
        raise ValueError("Required M-Schema keys exceed the schema prompt budget")
    return complete, {"schema_format": M_SCHEMA_VERSION, "prompt_bytes": len(complete.encode("utf-8")),
                      "rendered_columns": [{"table": row["table"], "column": row["column"]} for row in kept]}
