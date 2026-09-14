"""V12 column notes: descriptions, stored-value examples, and storage formats for prompt grounding."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .value_profiles import ValueProfiler


NOTES_VERSION = "sql-v12-column-notes-v1"
TEXT_FORMATS = {"currency_text", "thousands_separator", "percentage_text", "plus_suffix",
                "date_text", "time_text", "numeric_text"}


def _read_rows(path: Path) -> list[dict]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except (UnicodeError, csv.Error):
            continue
    return []


def load_descriptions(database: Path) -> dict[tuple[str, str], str]:
    """Read BIRD database_description CSVs stored next to the resolved database file."""
    folder = database.resolve().parent / "database_description"
    notes: dict[tuple[str, str], str] = {}
    if not folder.is_dir():
        return notes
    for path in sorted(folder.glob("*.csv")):
        for raw in _read_rows(path):
            row = {(key or "").strip().lower(): (value or "").strip() for key, value in raw.items()}
            name = row.get("original_column_name") or row.get("column_name")
            if not name:
                continue
            text = " ".join(" ".join(part.split()) for part in (row.get("column_description", ""),
                                                                 row.get("value_description", "")) if part)
            if text:
                notes[(path.stem.casefold(), name.casefold())] = text[:160]
    return notes


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def column_notes(database: Path, schema: list[dict], question: str, profiler: ValueProfiler,
                 *, byte_budget: int = 6000) -> tuple[str, dict]:
    """Render per-column notes from database files only; reference SQL is never consulted."""
    descriptions = load_descriptions(database)
    profile = profiler.build(database, schema)
    stats_by_column = {(row["table"], row["column"]): row for row in profile["columns"]
                       if row.get("status") == "complete"}
    terms = set(re.findall(r"[a-z0-9]+", question.casefold().replace("_", " ")))
    entries = []
    for table_order, table in enumerate(schema):
        table_name = table["table"]
        for column in table.get("column_details", []):
            name = column["name"]
            parts, priority = [], 0
            description = descriptions.get((table_name.casefold(), name.casefold()), "")
            if description and _normalized(description) != _normalized(name):
                parts.append(description)
            stats = stats_by_column.get((table_name, name), {})
            holds_text = stats.get("sqlite_types", {}).get("text", 0) > 0
            formats = sorted(set(stats.get("format_tags", [])) & TEXT_FORMATS) if holds_text else []
            if formats:
                parts.append("stored as text: " + ", ".join(formats))
                priority += 3
            if holds_text and stats.get("sample_values"):
                parts.append("examples: " + ", ".join(repr(value[:40]) for value in stats["sample_values"][:3]))
            if not parts:
                continue
            words = set(re.findall(r"[a-z0-9]+", f"{table_name} {name} {description}".casefold().replace("_", " ")))
            priority += 2 * len(terms & words)
            entries.append({"order": (table_order, len(entries)), "priority": priority,
                            "line": f"- {table_name}.{name}: " + " | ".join(parts)})
    header = "### Column notes"
    kept, used = [], len(header.encode("utf-8")) + 1
    for entry in sorted(entries, key=lambda row: (-row["priority"], row["order"])):
        size = len(entry["line"].encode("utf-8")) + 1
        if used + size <= byte_budget:
            kept.append(entry)
            used += size
    kept.sort(key=lambda row: row["order"])
    text = "\n".join([header, *(row["line"] for row in kept)]) if kept else ""
    return text, {"notes_version": NOTES_VERSION, "notes_bytes": len(text.encode("utf-8")),
                  "notes_columns": len(kept), "notes_candidates": len(entries),
                  "value_profile_identity": profile["identity"]}


V13_NOTES_VERSION = "sql-v13-column-notes-v1"
_SIMPLE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOPWORDS = {"the", "and", "for", "with", "what", "which", "how", "are", "was", "were", "that", "this", "from",
              "who", "has", "have", "refers", "refer", "benchmark", "evidence", "among", "all", "any", "their",
              "they", "its"}


def sql_identifier(name: str) -> str:
    """Render an identifier exactly as it must be written in SQLite."""
    return name if _SIMPLE_IDENTIFIER.fullmatch(name) else '"' + name.replace('"', '""') + '"'


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", text.casefold().replace("_", " "))
            if len(term) >= 3 and term not in _STOPWORDS}


def column_notes_v13(database: Path, schema: list[dict], question: str, profiler: ValueProfiler,
                     *, byte_budget: int = 3000) -> tuple[str, dict]:
    """Question-relevant notes with SQL-ready identifiers and no table-qualified column names."""
    descriptions = load_descriptions(database)
    profile = profiler.build(database, schema)
    stats_by_column = {(row["table"], row["column"]): row for row in profile["columns"]
                       if row.get("status") == "complete"}
    lowered, terms = question.casefold(), _terms(question)
    entries = []
    for table_order, table in enumerate(schema):
        table_name = table["table"]
        for column in table.get("column_details", []):
            name = column["name"]
            stats = stats_by_column.get((table_name, name), {})
            description = descriptions.get((table_name.casefold(), name.casefold()), "")
            holds_text = stats.get("sqlite_types", {}).get("text", 0) > 0
            matches = [value for value in stats.get("categorical_values", []) if holds_text and len(value) >= 2 and
                       re.search(r"(?<![a-z0-9])" + re.escape(value.casefold()) + r"(?![a-z0-9])", lowered)][:3]
            relevance = len(terms & _terms(f"{name} {description}")) + 3 * len(matches)
            if relevance == 0:
                continue
            parts = []
            if description and _normalized(description) != _normalized(name):
                parts.append(description)
            formats = sorted(set(stats.get("format_tags", [])) & TEXT_FORMATS) if holds_text else []
            if formats:
                parts.append("stored as text: " + ", ".join(formats))
            if matches:
                parts.append("stored values matching the question: " + ", ".join(repr(value[:40]) for value in matches))
            elif holds_text and stats.get("sample_values"):
                parts.append("examples: " + ", ".join(repr(value[:40]) for value in stats["sample_values"][:3]))
            if not parts:
                continue
            line = f"- {sql_identifier(name)} in table {sql_identifier(table_name)}: " + " | ".join(parts)
            entries.append({"order": (table_order, len(entries)), "priority": relevance + 2 * bool(formats),
                            "line": line})
    header = "### Column notes (identifiers are written exactly as they must appear in SQL)"
    kept, used = [], len(header.encode("utf-8")) + 1
    for entry in sorted(entries, key=lambda row: (-row["priority"], row["order"])):
        size = len(entry["line"].encode("utf-8")) + 1
        if used + size <= byte_budget:
            kept.append(entry)
            used += size
    kept.sort(key=lambda row: row["order"])
    text = "\n".join([header, *(row["line"] for row in kept)]) if kept else ""
    return text, {"notes_version": V13_NOTES_VERSION, "notes_bytes": len(text.encode("utf-8")),
                  "notes_columns": len(kept), "notes_candidates": len(entries),
                  "notes_truncated": len(entries) - len(kept), "value_profile_identity": profile["identity"]}
