"""Deterministic, bounded, read-only semantic probes for V9."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .guardrails import authorizer
from .v9_reasoning import ScopedEvidenceFact, ScopedEvidencePacket


PROBE_VERSION = "sql-v9-semantic-probes-v1"


class ProbeObservation(BaseModel):
    probe_id: str
    evidence_id: str
    kind: str
    verified: bool
    observation: dict = Field(default_factory=dict)
    elapsed_seconds: float


@dataclass(frozen=True)
class VerifiedEvidencePacket:
    identity: str
    facts: list[ScopedEvidenceFact]
    observations: list[ProbeObservation]
    text: str
    database_hash_unchanged: bool


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execute(connection: sqlite3.Connection, sql: str, params: tuple, *, timeout: float) -> tuple:
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    try:
        return connection.execute(sql, params).fetchone() or tuple()
    finally:
        connection.set_progress_handler(None, 0)


def verify(database: Path, packet: ScopedEvidencePacket, *, max_probes: int = 4,
           timeout_seconds: float = 1.0) -> VerifiedEvidencePacket:
    if not 0 <= max_probes <= 4:
        raise ValueError("V9 permits zero to four deterministic probes")
    before = _file_sha256(database)
    observations: list[ProbeObservation] = []
    updated: list[ScopedEvidenceFact] = []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.enable_load_extension(False)
        connection.set_authorizer(authorizer)
        for scoped in packet.facts:
            fact = scoped.fact
            observation = None
            if len(observations) < max_probes and fact.db_value is not None and fact.column:
                started = time.perf_counter()
                sql = (f"SELECT COUNT(*) FROM {_quoted(fact.table)} "
                       f"WHERE {_quoted(fact.column)} = ?")
                try:
                    row = _execute(connection, sql, (fact.db_value,), timeout=timeout_seconds)
                    count = int(row[0]) if row else 0
                    verified = count > 0
                    outcome = {"match_count": count, "mapping": f"{fact.table}.{fact.column}",
                               "printed_value": fact.db_value}
                except sqlite3.DatabaseError as error:
                    verified = False; outcome = {"error": type(error).__name__}
                observation = ProbeObservation(
                    probe_id=f"P{len(observations) + 1}", evidence_id=scoped.evidence_id,
                    kind="value_existence", verified=verified, observation=outcome,
                    elapsed_seconds=time.perf_counter() - started)
                observations.append(observation)
            elif len(observations) < max_probes and fact.type == "numeric_or_storage" and fact.column:
                started = time.perf_counter()
                sql = (f"SELECT COUNT(*), SUM(typeof({_quoted(fact.column)}) IN ('integer','real')), "
                       f"MIN(CAST({_quoted(fact.column)} AS REAL)), MAX(CAST({_quoted(fact.column)} AS REAL)) "
                       f"FROM {_quoted(fact.table)} WHERE {_quoted(fact.column)} IS NOT NULL")
                try:
                    row = _execute(connection, sql, tuple(), timeout=timeout_seconds)
                    outcome = {"non_null_count": row[0], "native_numeric_count": row[1],
                               "numeric_min": row[2], "numeric_max": row[3]}
                    verified = bool(row and row[0])
                except sqlite3.DatabaseError as error:
                    verified = False; outcome = {"error": type(error).__name__}
                observation = ProbeObservation(
                    probe_id=f"P{len(observations) + 1}", evidence_id=scoped.evidence_id,
                    kind="numeric_storage", verified=verified, observation=outcome,
                    elapsed_seconds=time.perf_counter() - started)
                observations.append(observation)
            elif len(observations) < max_probes and fact.type == "join_relationship" and fact.column:
                started = time.perf_counter()
                target_table, target_column = (fact.support.get("target_table"), fact.support.get("target_column"))
                if target_table and target_column:
                    sql = (f"SELECT COUNT(*) FROM {_quoted(fact.table)} AS s JOIN {_quoted(str(target_table))} AS t "
                           f"ON s.{_quoted(fact.column)} = t.{_quoted(str(target_column))}")
                    try:
                        row = _execute(connection, sql, tuple(), timeout=timeout_seconds)
                        outcome = {"joined_row_count": int(row[0]) if row else 0,
                                   "join": f"{fact.table}.{fact.column}={target_table}.{target_column}"}
                        verified = bool(row and row[0])
                    except sqlite3.DatabaseError as error:
                        verified = False; outcome = {"error": type(error).__name__}
                else:
                    verified = False; outcome = {"error": "missing_target"}
                observation = ProbeObservation(
                    probe_id=f"P{len(observations) + 1}", evidence_id=scoped.evidence_id,
                    kind="join_cardinality", verified=verified, observation=outcome,
                    elapsed_seconds=time.perf_counter() - started)
                observations.append(observation)
            updated.append(scoped.model_copy(update={"verified": bool(observation and observation.verified),
                                                       "verification_id": observation.probe_id if observation else None}))
    unchanged = before == _file_sha256(database)
    if not unchanged:
        raise RuntimeError("Semantic probe changed the database byte hash")
    lines = []
    by_id = {item.evidence_id: item for item in observations}
    for scoped in updated:
        fact = scoped.fact
        mapping = f"{fact.table}.{fact.column}" if fact.column else fact.table
        if fact.db_value is not None:
            mapping += f" = {json.dumps(fact.db_value, ensure_ascii=False)}"
        suffix = ""
        if scoped.evidence_id in by_id:
            item = by_id[scoped.evidence_id]
            suffix = (f" Verified observation: {json.dumps(item.observation, ensure_ascii=False)}."
                      if item.verified else " Probe did not verify this mapping; do not rely on it.")
        lines.append(f"[{scoped.evidence_id}] type={fact.type}; concept={json.dumps(fact.phrase)}; "
                     f"mapping={mapping}; allowed={','.join(scoped.allowed_effects)}; "
                     f"forbidden={','.join(scoped.forbidden_effects)}. {scoped.instruction}{suffix}")
    payload = {"version": PROBE_VERSION, "source": packet.identity,
               "observations": [row.model_dump() for row in observations]}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return VerifiedEvidencePacket(identity=identity, facts=updated, observations=observations,
                                  text="\n".join(lines), database_hash_unchanged=unchanged)
