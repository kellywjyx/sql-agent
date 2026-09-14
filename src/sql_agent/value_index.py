"""Question-conditioned, read-only value grounding for SQL Agent V7."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .execution import execute_sql
from .guardrails import authorizer, validate
from .schema_retrieval import schema_documents, tokens


VALUE_INDEX_VERSION = "sql-value-index-v1"
EVIDENCE_PACKET_VERSION = "sql-evidence-packet-v1"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def normalize_value(value: object) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def _interpret(value: str) -> tuple[str | None, float | None, str | None]:
    text = value.strip()
    currency = re.fullmatch(r"([$£€¥])\s*([+-]?[\d,]+(?:\.\d+)?)", text)
    percentage = re.fullmatch(r"([+-]?[\d,.]+)\s*%", text)
    numeric = re.fullmatch(r"[+-]?[\d,]+(?:\.\d+)?", text)
    date = re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?", text)
    try:
        match = currency or percentage or numeric
        raw_number = match.group(2) if currency else match.group(1) if percentage else match.group(0) if numeric else None
        number = float(raw_number.replace(",", "")) if raw_number is not None else None
    except (ValueError, AttributeError):
        number = None
    if currency:
        return "currency", number, None
    if percentage:
        return "percentage", number, None
    if numeric:
        return "numeric_text", number, None
    if date:
        return "date", None, text
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?", text):
        return "time", None, text
    return None, None, None


def _identity(database: Path, schema: list[dict]) -> str:
    documents = schema_documents(database, schema)
    basis = json.dumps({"version": VALUE_INDEX_VERSION, "schema": schema, "documents": documents},
                       sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    digest.update(hashlib.sha256(database.read_bytes()).digest())
    digest.update(basis)
    return digest.hexdigest()


def question_phrases(question: str) -> list[str]:
    natural = question.split("Benchmark evidence:", 1)[0]
    phrases: list[str] = []
    phrases.extend(re.findall(r"['\"]([^'\"\n]{1,120})['\"]", natural))
    phrases.extend(re.findall(r"[$£€¥]?[+-]?[\d][\d,.]*(?:\s*%)?|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", natural))
    phrases.extend(re.findall(r"\b(?:[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,3})\b", natural))
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9_.-]+", natural)
             if word.casefold() not in {"what", "which", "where", "when", "whose", "show", "list", "give",
                                        "find", "return", "with", "from", "that", "this", "have", "has", "had",
                                        "each", "every", "their", "there", "than", "then", "into", "about"}]
    phrases.extend(words)
    phrases.extend(" ".join(words[index:index + width]) for width in (2, 3)
                   for index in range(max(0, len(words) - width + 1)))
    unique = []
    for phrase in phrases:
        phrase = phrase.strip()
        if len(phrase) >= 2 and normalize_value(phrase) not in {normalize_value(item) for item in unique}:
            unique.append(phrase)
    return unique[:80]


@dataclass(frozen=True)
class EvidencePacket:
    identity: str
    text: str
    facts: list[dict]
    phrases: list[str]
    probe_count: int
    byte_count: int


class ValueIndex:
    """Stores bounded source-derived profiles in a separate SQLite artifact."""

    def __init__(self, artifacts: Path):
        self.root = artifacts.resolve() / "sql-agent" / "value-index"

    def _path(self, identity: str) -> Path:
        return self.root / identity / "values.sqlite"

    def build(self, database: Path, schema: list[dict], *, force: bool = False) -> dict:
        database = database.resolve(strict=True)
        identity = _identity(database, schema)
        target = self._path(identity)
        manifest = target.with_name("manifest.json")
        if target.exists() and manifest.exists() and not force:
            return json.loads(manifest.read_text(encoding="utf-8"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        if temporary.exists():
            temporary.unlink()
        started = perf_counter()
        documents = schema_documents(database, schema)
        descriptions = {(row.get("table"), row.get("column")): row.get("description", "")
                        for row in documents if row.get("kind") == "column"}
        source_connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        source_connection.execute("PRAGMA query_only=ON")
        source_connection.enable_load_extension(False)
        source_connection.set_authorizer(authorizer)

        def source_rows(sql: str, timeout: float = 1):
            validate(sql)
            deadline = time.monotonic() + timeout
            source_connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            try:
                return source_connection.execute(sql).fetchall()
            except sqlite3.OperationalError as error:
                if "interrupted" in str(error).casefold():
                    raise TimeoutError("Read-only value-index query exceeded its deadline") from error
                raise ValueError(str(error)) from error
            finally:
                source_connection.set_progress_handler(None, 0)
        with sqlite3.connect(temporary) as output:
            output.executescript("""
                CREATE TABLE columns(
                    table_name TEXT NOT NULL, column_name TEXT NOT NULL, declared_type TEXT,
                    description TEXT, row_count INTEGER, null_count INTEGER, distinct_count INTEGER,
                    sqlite_types_json TEXT NOT NULL, numeric_min REAL, numeric_max REAL,
                    status TEXT NOT NULL, PRIMARY KEY(table_name, column_name));
                CREATE TABLE values_index(
                    table_name TEXT NOT NULL, column_name TEXT NOT NULL, original_value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL, frequency INTEGER NOT NULL, pattern TEXT,
                    numeric_value REAL, date_value TEXT,
                    PRIMARY KEY(table_name, column_name, original_value));
                CREATE INDEX value_normalized ON values_index(normalized_value);
                CREATE INDEX value_column ON values_index(table_name, column_name);
            """)
            for table in schema:
                table_name = table["table"]
                try:
                    row_count = int(source_rows(f"SELECT COUNT(*) FROM {_quote(table_name)}")[0][0])
                except (RuntimeError, TimeoutError, ValueError, IndexError, TypeError):
                    row_count = 0
                for column in table.get("column_details", []):
                    name = column["name"]
                    quoted = _quote(name)
                    source = _quote(table_name)
                    status = "complete"
                    try:
                        type_rows = source_rows(f"SELECT typeof({quoted}), COUNT(*) FROM {source} GROUP BY typeof({quoted})")
                        type_counts = {str(kind): int(count) for kind, count in type_rows}
                        null_count = type_counts.get("null", 0)
                        distinct = int(source_rows(f"SELECT COUNT(DISTINCT {quoted}) FROM {source}")[0][0])
                    except (RuntimeError, TimeoutError, ValueError, IndexError, TypeError):
                        type_counts, null_count, distinct, status = {}, 0, 0, "partial"
                    numeric_min = numeric_max = None
                    if type_counts and set(type_counts) <= {"null", "integer", "real"}:
                        try:
                            numeric_min, numeric_max = source_rows(
                                f"SELECT MIN({quoted}), MAX({quoted}) FROM {source}")[0]
                        except (RuntimeError, TimeoutError, ValueError, IndexError):
                            status = "partial"
                    output.execute("INSERT INTO columns VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                        table_name, name, column.get("type", ""), descriptions.get((table_name, name), "")[:500],
                        row_count, null_count, distinct, json.dumps(type_counts, sort_keys=True),
                        numeric_min, numeric_max, status))
                    value_limit = 100 if distinct <= 100 else 25
                    if distinct == 0:
                        continue
                    try:
                        rows = source_rows(
                            f"SELECT {quoted}, COUNT(*) AS frequency FROM {source} "
                            f"WHERE {quoted} IS NOT NULL GROUP BY {quoted} ORDER BY frequency DESC, {quoted} LIMIT {value_limit}",
                            timeout=1)
                    except (RuntimeError, TimeoutError, ValueError):
                        continue
                    for raw, frequency in rows:
                        if isinstance(raw, bytes):
                            continue
                        original = str(raw)[:300]
                        pattern, numeric_value, date_value = _interpret(original)
                        output.execute("INSERT OR IGNORE INTO values_index VALUES(?,?,?,?,?,?,?,?)", (
                            table_name, name, original, normalize_value(original), int(frequency), pattern,
                            numeric_value, date_value))
            output.commit()
        output.close()
        source_connection.close()
        if target.exists():
            target.unlink()
        temporary.replace(target)
        payload = {
            "version": VALUE_INDEX_VERSION, "identity": identity,
            "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "path": str(target), "build_seconds": perf_counter() - started,
        }
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def load(self, database: Path, schema: list[dict]) -> tuple[dict, Path]:
        identity = _identity(database.resolve(strict=True), schema)
        path = self._path(identity)
        manifest = path.with_name("manifest.json")
        if not path.exists() or not manifest.exists():
            raise RuntimeError("V7 value index missing. Run: sql-agent build-value-index --database <path> --artifacts <path>")
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("identity") != identity or value.get("version") != VALUE_INDEX_VERSION:
            raise RuntimeError("V7 value index identity differs from the current database or schema")
        return value, path

    def check(self, database: Path, schema: list[dict]) -> dict:
        manifest, _ = self.load(database, schema)
        return {"value_index": "ready", "value_index_identity": manifest["identity"],
                "value_index_version": manifest["version"]}

    def evidence_packet(self, database: Path, question: str, schema: list[dict], schema_selection: dict,
                        *, oracle_evidence: str = "", evidence_mode: str = "generated",
                        byte_budget: int = 4096, fact_limit: int = 12, max_probes: int = 4) -> EvidencePacket:
        if evidence_mode not in {"none", "oracle", "generated"}:
            raise ValueError("evidence_mode must be none, oracle, or generated")
        manifest, path = self.load(database, schema)
        phrases = question_phrases(question)
        selected = {(row["table"], row["column"]) for row in schema_selection.get("selected_columns", [])}
        facts: list[dict] = []
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            column_rows = connection.execute("SELECT * FROM columns").fetchall()
            value_rows = connection.execute("SELECT * FROM values_index").fetchall()
        phrase_norms = [(phrase, normalize_value(phrase)) for phrase in phrases]
        for row in value_rows:
            key = (row["table_name"], row["column_name"])
            if selected and key not in selected:
                continue
            matched = [original for original, normalized in phrase_norms
                       if normalized == row["normalized_value"] or
                       (len(normalized) >= 3 and normalized in row["normalized_value"]) or
                       (len(row["normalized_value"]) >= 3 and row["normalized_value"] in normalized)]
            if matched:
                facts.append({"kind": "value", "table": key[0], "column": key[1],
                              "printed_value": row["original_value"], "normalized_value": row["normalized_value"],
                              "frequency": row["frequency"], "pattern": row["pattern"],
                              "numeric_value": row["numeric_value"], "date_value": row["date_value"],
                              "matched_phrase": matched[0], "reason": "exact_or_normalized_value"})
        question_terms = set(tokens(question))
        for row in column_rows:
            key = (row["table_name"], row["column_name"])
            if selected and key not in selected:
                continue
            column_terms = set(tokens(f"{key[0]} {key[1]} {row['description'] or ''}"))
            overlap = sorted(question_terms & column_terms)
            if overlap:
                facts.append({"kind": "column", "table": key[0], "column": key[1],
                              "description": row["description"] or "", "declared_type": row["declared_type"],
                              "sqlite_types": json.loads(row["sqlite_types_json"]),
                              "numeric_min": row["numeric_min"], "numeric_max": row["numeric_max"],
                              "reason": "question_description_overlap", "matched_terms": overlap})
        for path_row in schema_selection.get("join_paths", [])[:4]:
            facts.append({"kind": "join_path", "tables": path_row, "reason": "foreign_key_closure"})
        probes = 0
        found_phrases = {normalize_value(row.get("matched_phrase", "")) for row in facts}
        unresolved = [phrase for phrase in phrases if normalize_value(phrase) not in found_phrases]
        text_columns = [(table["table"], column["name"]) for table in schema
                        for column in table.get("column_details", [])
                        if any(kind in (column.get("type") or "").upper() for kind in ("TEXT", "CHAR", "CLOB"))
                        and (not selected or (table["table"], column["name"]) in selected)]
        for phrase in unresolved:
            if probes >= max_probes or len(facts) >= fact_limit:
                break
            for table_name, column_name in text_columns:
                if probes >= max_probes:
                    break
                probes += 1
                literal = phrase.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
                try:
                    rows = execute_sql(database,
                        f"SELECT {_quote(column_name)}, COUNT(*) FROM {_quote(table_name)} "
                        f"WHERE {_quote(column_name)} LIKE '%{literal}%' ESCAPE '\\' "
                        f"GROUP BY {_quote(column_name)} ORDER BY COUNT(*) DESC LIMIT 5",
                        timeout=1, max_rows=5)["rows"]
                except (RuntimeError, TimeoutError, ValueError):
                    continue
                for raw, frequency in rows:
                    facts.append({"kind": "value", "table": table_name, "column": column_name,
                                  "printed_value": str(raw)[:300], "normalized_value": normalize_value(raw),
                                  "frequency": frequency, "matched_phrase": phrase,
                                  "reason": "bounded_read_only_probe"})
                if rows:
                    break
        if evidence_mode == "oracle" and oracle_evidence:
            facts.insert(0, {"kind": "oracle_benchmark_evidence", "text": oracle_evidence[:1200],
                             "reason": "diagnostic_only"})
        elif evidence_mode == "none":
            facts = []
        priority = {"oracle_benchmark_evidence": 0, "value": 1, "join_path": 2, "column": 3}
        facts.sort(key=lambda row: (priority.get(row["kind"], 9), row.get("table", ""), row.get("column", "")))
        kept: list[dict] = []
        for fact in facts:
            candidate = [*kept, fact]
            text = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(candidate) <= fact_limit and len(text.encode("utf-8")) <= byte_budget:
                kept.append(fact)
        text = json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
        identity_basis = {"version": EVIDENCE_PACKET_VERSION, "value_index": manifest["identity"],
                          "question": question.split("Benchmark evidence:", 1)[0], "mode": evidence_mode,
                          "facts": kept}
        identity = hashlib.sha256(json.dumps(identity_basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return EvidencePacket(identity, text, kept, phrases, probes, len(text.encode("utf-8")))
