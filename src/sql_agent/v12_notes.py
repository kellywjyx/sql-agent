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
