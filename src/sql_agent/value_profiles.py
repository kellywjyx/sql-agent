"""Bounded, read-only SQLite value profiles stored outside versioned source."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .execution import execute_sql


PROFILE_VERSION = "sql-value-profile-v2"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _identity(database: Path, schema: list[dict]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    digest = hashlib.sha256()
    digest.update(PROFILE_VERSION.encode())
    digest.update(hashlib.sha256(database.read_bytes()).digest())
    digest.update(canonical)
    return digest.hexdigest()


def _format_tags(values: list[str]) -> list[str]:
    tags: set[str] = set()
    for value in values:
        text = value.strip()
        if re.fullmatch(r"[$£€¥]\s*[+-]?[\d,]+(?:\.\d+)?", text):
            tags.add("currency_text")
        if re.fullmatch(r"[+-]?[\d]{1,3}(?:,\d{3})+(?:\.\d+)?\+?", text):
            tags.add("thousands_separator")
        if re.fullmatch(r"[+-]?[\d,.]+%", text):
            tags.add("percentage_text")
        if re.fullmatch(r"[\d,]+\+", text):
            tags.add("plus_suffix")
        if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?", text):
            tags.add("date_text")
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?", text):
            tags.add("time_text")
        if re.fullmatch(r"[+-]?[\d,.]+", text) and not re.search(r"[A-Za-z]", text):
            tags.add("numeric_text")
    return sorted(tags)


class ValueProfiler:
    """Creates bounded profiles through the same isolated read-only SQL worker."""

    def __init__(self, artifacts: Path):
        self.root = artifacts.resolve() / "sql-agent" / "value-profiles"

    def _path(self, identity: str) -> Path:
        return self.root / identity / "profile.json"

    def build(self, database: Path, schema: list[dict]) -> dict:
        database = database.resolve(strict=True)
        identity = _identity(database, schema)
        target = self._path(identity)
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        columns: list[dict] = []
        for table in schema:
            table_name = table["table"]
            details = table.get("column_details", [])
            try:
                total = int(execute_sql(database, f"SELECT COUNT(*) FROM {_quote(table_name)}",
                                        timeout=1, max_rows=1)["rows"][0][0])
            except (RuntimeError, TimeoutError, ValueError):
                total = 0
            for start in range(0, len(details), 32):
                chunk = details[start:start + 32]
                try:
                    sample_rows = execute_sql(database,
                        "SELECT " + ", ".join(_quote(column["name"]) for column in chunk) +
                        f" FROM {_quote(table_name)} LIMIT 100", timeout=1, max_rows=100)["rows"]
                except (RuntimeError, TimeoutError, ValueError):
                    columns.extend({"table": table_name, "column": column["name"], "status": "unavailable"}
                                   for column in chunk)
                    continue
                for offset, column in enumerate(chunk):
                    raw = [row[offset] for row in sample_rows]
                    nonnull = [value for value in raw if value is not None]
                    samples = list(dict.fromkeys(str(value)[:160] for value in nonnull))
                    type_counts = {}
                    for value in raw:
                        kind = "null" if value is None else "integer" if isinstance(value, int) else \
                               "real" if isinstance(value, float) else "blob" if isinstance(value, bytes) else "text"
                        type_counts[kind] = type_counts.get(kind, 0) + 1
                    columns.append({
                        "table": table_name, "column": column["name"], "declared_type": column.get("type", ""),
                        "row_count": total, "sample_size": len(raw),
                        "null_fraction": (sum(value is None for value in raw) / len(raw) if raw else 0.0),
                        "distinct_count": len(samples), "distinct_count_is_sampled": total > len(raw),
                        "sqlite_types": type_counts, "format_tags": _format_tags(samples),
                        "categorical_values": samples[:25] if len(samples) <= 100 else [],
                        "sample_values": samples[:5], "status": "complete",
                    })
        payload = {"version": PROFILE_VERSION, "identity": identity,
                   "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(), "columns": columns}
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        return payload

    def load(self, database: Path, schema: list[dict]) -> dict:
        identity = _identity(database.resolve(strict=True), schema)
        target = self._path(identity)
        if not target.exists():
            raise RuntimeError("Value profile missing. Run: sql-agent profile-values --database <path> --artifacts <path>")
        return json.loads(target.read_text(encoding="utf-8"))

    def check(self, database: Path, schema: list[dict]) -> dict:
        profile = self.load(database, schema)
        return {"value_profile_identity": profile["identity"], "value_profile_version": profile["version"]}

    @staticmethod
    def relevant(profile: dict, question: str, selected_columns: list[dict], limit: int = 12) -> list[dict]:
        terms = set(re.findall(r"[a-z0-9]+", question.casefold().replace("_", " ")))
        allowed = {(row["table"], row["column"]) for row in selected_columns}
        ranked = []
        for row in profile.get("columns", []):
            if allowed and (row.get("table"), row.get("column")) not in allowed:
                continue
            name_terms = set(re.findall(r"[a-z0-9]+", f"{row.get('table','')} {row.get('column','')}".casefold().replace("_", " ")))
            value_match = [value for value in row.get("categorical_values", [])
                           if any(term and term in value.casefold() for term in terms)]
            score = 4 * len(terms & name_terms) + 3 * bool(value_match) + bool(row.get("format_tags"))
            if score:
                ranked.append((score, row, value_match[:5]))
        ranked.sort(key=lambda item: (-item[0], item[1].get("table", ""), item[1].get("column", "")))
        return [{"table": row["table"], "column": row["column"],
                 "format_tags": row.get("format_tags", []), "sqlite_types": row.get("sqlite_types", {}),
                 "distinct_count": row.get("distinct_count"),
                 "values": matches or row.get("sample_values", [])[:3]}
                for _, row, matches in ranked[:limit]]
