"""Deterministic database-side knowledge cache for V8 evidence reconstruction."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from time import perf_counter

from .guardrails import authorizer, validate
from .schema_retrieval import _description_rows
from .value_index import _interpret, normalize_value


CACHE_VERSION = "sql-v8-knowledge-cache-v2"
NORMALIZATION_VERSION = "sql-v8-normalization-v1"


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def char_ngrams(value: str, size: int = 3) -> list[str]:
    compact = re.sub(r"[^a-z0-9]+", "", normalize_value(value))
    if len(compact) < size:
        return [compact] if compact else []
    return sorted({compact[index:index + size] for index in range(len(compact) - size + 1)})


def word_tokens(value: str) -> list[str]:
    return sorted(set(re.findall(r"[a-z0-9]+", normalize_value(value))))


def cache_identity(database: Path, schema: list[dict]) -> str:
    descriptions = _description_rows(database)
    canonical_descriptions = [{"table": table, "column": column, **value}
                              for (table, column), value in sorted(descriptions.items())]
    basis = {"version": CACHE_VERSION, "normalization": NORMALIZATION_VERSION,
             "schema": schema, "descriptions": canonical_descriptions}
    digest = hashlib.sha256(database.read_bytes())
    digest.update(json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str).encode())
    return digest.hexdigest()


class KnowledgeCache:
    """Whole-column profiles and bounded value indexes, never written into source DBs."""

    def __init__(self, artifacts: Path):
        self.root = artifacts.resolve() / "sql-agent/knowledge-cache"

    def _path(self, identity: str) -> Path:
        return self.root / identity / "knowledge.sqlite"

    def build(self, database: Path, schema: list[dict], *, force: bool = False,
              query_timeout: float = 2.0) -> dict:
        database = database.resolve(strict=True)
        identity = cache_identity(database, schema)
        target = self._path(identity)
        manifest_path = target.with_name("manifest.json")
        if target.is_file() and manifest_path.is_file() and not force:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        descriptions = _description_rows(database)
        started = perf_counter()
        source = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        source.execute("PRAGMA query_only=ON")
        source.enable_load_extension(False)
        source.set_authorizer(authorizer)

        def rows(sql: str, parameters=()):
            validate(sql)
            deadline = time.monotonic() + query_timeout
            source.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            try:
                return source.execute(sql, parameters).fetchall()
            finally:
                source.set_progress_handler(None, 0)

        partial_columns = 0
        with sqlite3.connect(temporary) as output:
            output.executescript("""
                CREATE TABLE columns(
                    table_name TEXT NOT NULL, column_name TEXT NOT NULL, declared_type TEXT,
                    display_name TEXT, description TEXT, value_description TEXT,
                    row_count INTEGER, null_count INTEGER, distinct_count INTEGER,
                    sqlite_types_json TEXT, numeric_min REAL, numeric_max REAL,
                    numeric_text_rate REAL, date_rate REAL, percentage_rate REAL, currency_rate REAL,
                    categorical INTEGER NOT NULL, status TEXT NOT NULL,
                    PRIMARY KEY(table_name,column_name));
                CREATE TABLE values_index(
                    table_name TEXT NOT NULL, column_name TEXT NOT NULL, original_value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL, frequency INTEGER NOT NULL, pattern TEXT,
                    numeric_value REAL, indexed_scope TEXT NOT NULL,
                    PRIMARY KEY(table_name,column_name,original_value));
                CREATE TABLE value_terms(
                    term_type TEXT NOT NULL, term TEXT NOT NULL, table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL, original_value TEXT NOT NULL);
                CREATE INDEX value_normalized ON values_index(normalized_value);
                CREATE INDEX value_column ON values_index(table_name,column_name);
                CREATE INDEX value_term ON value_terms(term_type,term);
                CREATE TABLE relationships(
                    source_table TEXT, source_column TEXT, target_table TEXT, target_column TEXT,
                    PRIMARY KEY(source_table,source_column,target_table,target_column));
            """)
            for table in schema:
                table_name = table["table"]
                for fk in table.get("foreign_keys", []):
                    output.execute("INSERT OR IGNORE INTO relationships VALUES(?,?,?,?)",
                                   (table_name, fk["from"], fk["table"], fk.get("to")))
                try:
                    row_count = int(rows(f"SELECT COUNT(*) FROM {quote(table_name)}")[0][0])
                except (sqlite3.DatabaseError, TimeoutError, IndexError, TypeError):
                    row_count = 0
                for column in table.get("column_details", []):
                    name, status = column["name"], "complete"
                    qtable, qcolumn = quote(table_name), quote(name)
                    extra = descriptions.get((table_name.casefold(), name.casefold()), {})
                    try:
                        types = {str(kind): int(count) for kind, count in rows(
                            f"SELECT typeof({qcolumn}), COUNT(*) FROM {qtable} GROUP BY typeof({qcolumn})")}
                        null_count = types.get("null", 0)
                        distinct_count = int(rows(f"SELECT COUNT(DISTINCT {qcolumn}) FROM {qtable}")[0][0])
                    except (sqlite3.DatabaseError, TimeoutError, IndexError, TypeError):
                        types, null_count, distinct_count, status = {}, 0, 0, "partial"
                    numeric_min = numeric_max = None
                    if types and set(types) <= {"null", "integer", "real"}:
                        try:
                            numeric_min, numeric_max = rows(f"SELECT MIN({qcolumn}), MAX({qcolumn}) FROM {qtable}")[0]
                        except (sqlite3.DatabaseError, TimeoutError, IndexError):
                            status = "partial"
                    rates = {"numeric_text": 0.0, "date": 0.0, "percentage": 0.0, "currency": 0.0}
                    value_rows = []
                    if distinct_count:
                        limit = distinct_count if distinct_count <= 100 else 25
                        scope = "all_distinct" if distinct_count <= 100 else "top_frequency"
                        try:
                            value_rows = rows(
                                f"SELECT {qcolumn}, COUNT(*) FROM {qtable} WHERE {qcolumn} IS NOT NULL "
                                f"GROUP BY {qcolumn} ORDER BY COUNT(*) DESC, CAST({qcolumn} AS TEXT) LIMIT {int(limit)}")
                        except (sqlite3.DatabaseError, TimeoutError):
                            status = "partial"
                    pattern_counts = {key: 0 for key in rates}
                    observed = 0
                    for raw, frequency in value_rows:
                        if raw is None or isinstance(raw, bytes):
                            continue
                        original = str(raw)[:500]
                        # Storage interpretation applies to printed TEXT. Native
                        # INTEGER/REAL values must never be advertised as needing
                        # a CAST merely because profiling converted them to str.
                        pattern, number, _ = _interpret(original) if isinstance(raw, str) else (None, None, None)
                        observed += int(frequency)
                        if pattern in pattern_counts:
                            pattern_counts[pattern] += int(frequency)
                        output.execute("INSERT OR IGNORE INTO values_index VALUES(?,?,?,?,?,?,?,?)",
                                       (table_name, name, original, normalize_value(original), int(frequency),
                                        pattern, number, scope))
                        for token in word_tokens(original):
                            output.execute("INSERT INTO value_terms VALUES('token',?,?,?,?)",
                                           (token, table_name, name, original))
                        for ngram in char_ngrams(original):
                            output.execute("INSERT INTO value_terms VALUES('trigram',?,?,?,?)",
                                           (ngram, table_name, name, original))
                    if observed:
                        rates = {key: count / observed for key, count in pattern_counts.items()}
                    categorical = int(0 < distinct_count <= 100)
                    if status == "partial":
                        partial_columns += 1
                    output.execute("INSERT INTO columns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        table_name, name, column.get("type", ""), extra.get("display_name", "")[:300],
                        extra.get("description", "")[:2000], extra.get("value_description", "")[:2000],
                        row_count, null_count, distinct_count, json.dumps(types, sort_keys=True),
                        numeric_min, numeric_max, rates["numeric_text"], rates["date"],
                        rates["percentage"], rates["currency"], categorical, status))
            output.commit()
        output.close()
        source.close()
        after = hashlib.sha256(database.read_bytes()).hexdigest()
        if before != after:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Source database hash changed while building the read-only V8 cache")
        target.unlink(missing_ok=True)
        temporary.replace(target)
        manifest = {"version": CACHE_VERSION, "normalization_version": NORMALIZATION_VERSION,
                    "identity": identity, "database_sha256": before,
                    "cache_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "partial_columns": partial_columns, "build_seconds": perf_counter() - started}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def load(self, database: Path, schema: list[dict]) -> tuple[dict, Path]:
        identity = cache_identity(database.resolve(strict=True), schema)
        path, manifest_path = self._path(identity), self._path(identity).with_name("manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise RuntimeError("V8 knowledge cache missing. Run: sql-agent build-v8-knowledge --database <path> --artifacts <path>")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("version") != CACHE_VERSION:
            raise RuntimeError("V8 knowledge cache does not match the current database package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("cache_sha256"):
            raise RuntimeError("V8 knowledge cache checksum mismatch")
        return manifest, path
