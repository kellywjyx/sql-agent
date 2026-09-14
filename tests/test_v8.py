import hashlib
import json
import sqlite3

import pytest

from sql_agent.schema import inspect_schema
from sql_agent.v8_evaluation import evidence_metrics, oracle_gap_recovery, transitions
from sql_agent.v8_evidence import EvidenceReconstructor
from sql_agent.v8_knowledge import KnowledgeCache, cache_identity, char_ngrams


def knowledge_database(tmp_path):
    folder = tmp_path / "schools"
    folder.mkdir()
    database = folder / "schools.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schools(id INTEGER PRIMARY KEY, school_type TEXT, season TEXT, amount TEXT)")
        connection.executemany("INSERT INTO schools VALUES(?,?,?,?)", [
            (1, "PRI", "2015/2016", "$1,200"), (2, "PUB", "2014/2015", "900"),
            (3, "PUB", "2015/2016", "1,050")])
    descriptions = folder / "database_description"
    descriptions.mkdir()
    (descriptions / "schools.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "school_type,school type,ownership category,text,PRI means private school; PUB means public school\n"
        "season,season,competition season,text,season ends in the second printed year\n"
        "amount,amount,printed monetary amount,text,\n", encoding="utf-8")
    return database


def selection(cases):
    return [{"id": case.id, "group": "A"} for case in cases]


def test_v8_cache_profiles_complete_low_cardinality_and_preserves_source(tmp_path):
    database = knowledge_database(tmp_path)
    schema = inspect_schema(database)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    cache = KnowledgeCache(tmp_path / "artifacts")
    manifest = cache.build(database, schema)
    loaded, path = cache.load(database, schema)
    assert loaded["identity"] == manifest["identity"] == cache_identity(database, schema)
    with sqlite3.connect(path) as connection:
        values = connection.execute(
            "SELECT original_value,indexed_scope FROM values_index WHERE column_name='school_type' ORDER BY original_value").fetchall()
        profile = connection.execute(
            "SELECT row_count,distinct_count,categorical,value_description FROM columns WHERE column_name='school_type'").fetchone()
    assert values == [("PRI", "all_distinct"), ("PUB", "all_distinct")]
    assert profile[:3] == (3, 2, 1) and "private" in profile[3]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert char_ngrams("United States") == sorted(char_ngrams("United States"))


def test_v8_typed_routes_find_encoded_value_and_season_normalization(tmp_path):
    database = knowledge_database(tmp_path)
    schema = inspect_schema(database)
    artifacts = tmp_path / "artifacts"
    KnowledgeCache(artifacts).build(database, schema)
    packet = EvidenceReconstructor(artifacts).packet(
        database, "Which private schools competed in the 2016 season?", schema, mode="all")
    facts = [fact.model_dump() for fact in packet.facts]
    assert any(row["type"] == "encoded_value" and row["db_value"] == "PRI" for row in facts)
    assert any(row["type"] == "normalization" and row["db_value"] == "2015/2016" for row in facts)
    assert packet.byte_count <= 4096 and len(packet.facts) <= 12
    assert packet.probe_count <= 4
    assert "Benchmark evidence" not in packet.text


def test_v8_high_cardinality_exact_probe_is_bounded_and_description_is_not_overmapped(tmp_path):
    folder = tmp_path / "inventory"; folder.mkdir()
    database = folder / "inventory.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, code TEXT)")
        connection.executemany("INSERT INTO items VALUES(?,?)", [(i, f"ZZZ{i:03}") for i in range(150)])
    descriptions = folder / "database_description"; descriptions.mkdir()
    (descriptions / "items.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "code,item code,external item identifier,text,A means active; I means inactive\n", encoding="utf-8")
    schema = inspect_schema(database); artifacts = tmp_path / "artifacts"
    KnowledgeCache(artifacts).build(database, schema)
    packet = EvidenceReconstructor(artifacts).packet(database, "Find item code ZZZ149", schema)
    facts = [fact.model_dump() for fact in packet.facts]
    assert any(row.get("db_value") == "ZZZ149" and
               row["support"].get("scope") == "bounded_read_only_probe" for row in facts)
    assert packet.probe_count <= 4
    neutral = EvidenceReconstructor(artifacts).packet(database, "List item records", schema)
    assert not any(fact.type == "encoded_value" for fact in neutral.facts)
    assert not any(fact.transformation == "numeric_text_cast" for fact in neutral.facts)


def test_v8_native_numeric_columns_are_not_labeled_numeric_text(tmp_path):
    database = knowledge_database(tmp_path); schema = inspect_schema(database)
    artifacts = tmp_path / "artifacts"; KnowledgeCache(artifacts).build(database, schema)
    packet = EvidenceReconstructor(artifacts).packet(database, "List school records", schema)
    assert not any(fact.column == "id" and fact.transformation == "numeric_text_cast"
                   for fact in packet.facts)


def test_v8_cache_missing_is_actionable(tmp_path):
    database = knowledge_database(tmp_path)
    with pytest.raises(RuntimeError, match="build-v8-knowledge"):
        KnowledgeCache(tmp_path / "missing").load(database, inspect_schema(database))


class Case:
    def __init__(self, identifier, sql):
        self.id = identifier
        self.expected = {"sql": sql}


def test_v8_scoring_is_separate_and_measures_precision_mapping_and_gap():
    cases = [Case("a", "SELECT name FROM schools WHERE school_type = 'PRI'")]
    rows = [{"id": "a", "facts": [
        {"type": "encoded_value", "table": "schools", "column": "school_type", "db_value": "PRI"},
        {"type": "literal_value", "table": "schools", "column": "school_type", "db_value": "PUB"},
    ]}]
    report = evidence_metrics(cases, rows, selection(cases))
    assert report["literal_recall"] == 1
    assert report["evidence_precision"] == .5
    assert report["value_column_accuracy"] == 1
    assert oracle_gap_recovery(.3, .525, .435) == pytest.approx(.6)
    assert transitions({"a": False, "b": True}, {"a": True, "b": False}) == {
        "both_correct": 0, "wrong_to_correct": 1, "correct_to_wrong": 1, "both_wrong": 0}


def test_v8_source_contains_no_locked_final_or_oracle_runtime_paths():
    # The public V8 reconstructor has no evidence-mode escape hatch. Oracle text
    # exists only in the scoring-side decomposition artifact.
    import inspect
    source = inspect.getsource(EvidenceReconstructor.packet)
    assert "oracle_evidence" not in source
    assert "locked" not in source
