from pathlib import Path
import sqlite3

import pytest

from sql_agent.v8_evidence import EvidenceFact, StructuredEvidencePacket
from sql_agent.v9_data import prepare
from sql_agent.v9_probes import verify
from sql_agent.v9_reasoning import scope_packet
from sql_agent.v9_utilization import aggregate, analyze


def fact(kind="literal_value", *, value="PRI", column="school_type"):
    return EvidenceFact(phrase="private", type=kind, table="schools", column=column,
                        db_value=value, confidence=.95, source="test")


def packet(*facts):
    return StructuredEvidencePacket("source", "all", list(facts), "", 0, 0, 0)


def test_scoped_literal_only_allows_predicate():
    result = scope_packet(packet(fact()))
    row = result.facts[0]
    assert row.allowed_effects == ["predicate"]
    assert {"aggregation", "grouping", "projection"} <= set(row.forbidden_effects)
    assert "does not request" in result.text


def test_scoped_storage_explicitly_denies_aggregate_inference():
    result = scope_packet(packet(fact("numeric_or_storage", value=None, column="amount")))
    assert result.facts[0].allowed_effects == ["conversion"]
    assert "does not imply SUM" in result.text


def test_scoped_packet_obeys_byte_and_fact_limits():
    result = scope_packet(packet(*(fact(column=f"column_{index}") for index in range(20))),
                          fact_limit=12, byte_budget=700)
    assert len(result.facts) <= 12
    assert result.byte_count <= 700


def test_utilization_correct_ignored_contradicted_and_unnecessary():
    reference = "SELECT name FROM schools WHERE school_type = 'PRI'"
    correct = analyze(reference, "SELECT name FROM schools WHERE school_type = 'PRI'", [fact().model_dump()])
    ignored = analyze(reference, "SELECT name FROM schools", [fact().model_dump()])
    contradicted = analyze(reference, "SELECT name FROM schools WHERE school_type = 'PUB'", [fact().model_dump()])
    irrelevant = fact(value="CA", column="state").model_dump()
    unnecessary = analyze(reference, "SELECT name FROM schools WHERE state = 'CA'", [irrelevant])
    assert correct["facts"][0]["utilization"] == "correctly_used"
    assert ignored["facts"][0]["utilization"] == "ignored"
    assert contradicted["facts"][0]["utilization"] == "contradicted"
    assert unnecessary["facts"][0]["utilization"] == "unnecessarily_used"


def test_storage_utilization_requires_reference_conversion():
    storage = fact("numeric_or_storage", value=None, column="amount").model_dump()
    report = analyze("SELECT CAST(amount AS REAL) FROM sales",
                     "SELECT CAST(amount AS REAL) FROM sales", [storage])
    assert report["facts"][0]["utilization"] == "correctly_used"


def test_aggregate_utilization_metrics():
    reference = "SELECT name FROM schools WHERE school_type = 'PRI'"
    reports = [analyze(reference, reference, [fact().model_dump()]),
               analyze(reference, "SELECT name FROM schools", [fact().model_dump()])]
    assert aggregate(reports)["evidence_utilization_accuracy"] == .5


def test_probe_is_parameterized_read_only_and_hash_stable(tmp_path: Path):
    database = tmp_path / "demo.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schools(name TEXT, school_type TEXT)")
        connection.executemany("INSERT INTO schools VALUES (?, ?)", [("A", "PRI"), ("B", "PUB")])
    scoped = scope_packet(packet(fact()))
    result = verify(database, scoped)
    assert result.database_hash_unchanged
    assert result.observations[0].verified
    assert result.observations[0].observation["match_count"] == 1


def test_probe_rejects_more_than_four(tmp_path: Path):
    with pytest.raises(ValueError, match="zero to four"):
        verify(tmp_path / "missing.sqlite", scope_packet(packet()), max_probes=5)


def test_probe_numeric_storage_and_join_cardinality(tmp_path: Path):
    database = tmp_path / "relations.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY, amount REAL)")
        connection.execute("CREATE TABLE child(parent_id INTEGER)")
        connection.execute("INSERT INTO parent VALUES (1, 2.5)")
        connection.execute("INSERT INTO child VALUES (1)")
    storage = EvidenceFact(phrase="amount", type="numeric_or_storage", table="parent",
                           column="amount", confidence=.9, source="test")
    join = EvidenceFact(phrase="foreign-key path", type="join_relationship", table="child",
                        column="parent_id", confidence=.9, source="test",
                        support={"target_table": "parent", "target_column": "id"})
    result = verify(database, scope_packet(packet(storage, join)))
    assert [row.kind for row in result.observations] == ["numeric_storage", "join_cardinality"]
    assert all(row.verified for row in result.observations)


def test_protocol_preserves_locked_final(tmp_path: Path):
    v8 = tmp_path / "v8/sql-agent"
    v8.mkdir(parents=True)
    (v8 / "development-decision.json").write_text(
        '{"generated_evidence_ex":0.35}', encoding="utf-8")
    protocol = prepare(tmp_path)
    assert protocol["locked_final_opened"] is False
    assert "locked final" in protocol["forbidden"]
    assert protocol["public_default"] == "qwen_v5"
