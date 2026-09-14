import json
from pathlib import Path

from llm_evals import EvalCase
from llm_evals.campaign import freeze_dataset

from sql_agent.v10_data import prepare
from sql_agent.v10_correction import detect, localized_patch, retrieve
from sql_agent.v10_signature import build_signature


def test_v10_freezes_forty_uncaptured_cases_and_keeps_final_closed(tmp_path: Path):
    (tmp_path / "db.sqlite").write_bytes(b"fixture")
    source = tmp_path / "v7/sql-agent/data/calibration.jsonl"
    cases = [EvalCase(id=f"case-{index}", input=f"question {index}", expected={"sql": "SELECT 1"},
                      context={"database": str(tmp_path / "db.sqlite")},
                      metadata={"db_id": f"db-{index % 5}", "group_id": str(index)}) for index in range(60)]
    freeze_dataset(source, cases, "development")
    captured = tmp_path / "v9/sql-agent/eval/old/run"
    captured.mkdir(parents=True)
    (captured / "predictions.jsonl").write_text(json.dumps({"id": "case-0"}) + "\n", encoding="utf-8")
    for version, name in [("v3", "development"), ("v3", "final"), ("v5", "final")]:
        path = tmp_path / f"{version}/sql-agent/data/{name}.jsonl"
        freeze_dataset(path, cases[:1], "development")
    result = prepare(tmp_path)
    selection = json.loads((tmp_path / "v10/sql-agent/data/pilot-selection.json").read_text())
    assert result["locked_final_opened"] is False
    assert len(selection["ids"]) == 40
    assert "case-0" not in selection["ids"]


def test_operation_detector_abstains_or_removes_only_unrequested_aggregate():
    sql = "SELECT COUNT(name) FROM schools WHERE status = 'A'"
    detection = detect("List the school names with active status", sql, [], {"row_count": 1, "columns": ["COUNT(name)"]})
    assert detection.error_subtype == "unrequested_aggregate"
    patch = localized_patch(sql, detection)
    assert patch["sql"] == "SELECT name FROM schools WHERE status = 'A'"
    assert detect("How many schools are active?", "SELECT COUNT(*) FROM schools", []).abstained


def test_wrong_aggregate_and_grain_are_localized():
    operation = detect("How many students are there?", "SELECT AVG(age) FROM students", [])
    assert "COUNT(age)" in localized_patch("SELECT AVG(age) FROM students", operation)["sql"]
    grain_sql = "SELECT school, AVG(score) FROM scores"
    grain = detect("Show the average score for each school", grain_sql, [])
    patched = localized_patch(grain_sql, grain)["sql"]
    assert "GROUP BY school" in patched


def test_predicate_detector_uses_unique_scoped_mapping_and_preserves_query():
    facts = [{"phrase": "private", "type": "encoded_value", "table": "schools",
              "column": "school_type", "db_value": "P1"}]
    sql = "SELECT name FROM schools WHERE category = 'P1'"
    detection = detect("List private P1 schools", sql, facts)
    assert detection.error_type == "predicate"
    assert localized_patch(sql, detection)["sql"] == "SELECT name FROM schools WHERE schools.school_type = 'P1'"


def test_numeric_only_literal_column_repair_abstains():
    facts = [{"phrase": "45", "type": "literal_value", "table": "sales",
              "column": "store_id", "db_value": "45"}]
    assert detect("Find item 45", "SELECT item FROM sales WHERE item_id = 45", facts).abstained


def test_join_detector_requires_one_scoped_relationship():
    facts = [{"phrase": "foreign-key path", "type": "join_relationship", "table": "orders",
              "column": "customer_id", "support": {"target_table": "customers", "target_column": "id"}}]
    sql = "SELECT customers.name FROM orders JOIN customers ON orders.id = customers.id"
    detection = detect("List customer names for orders", sql, facts)
    assert detection.error_type == "join"
    assert "orders.customer_id = customers.id" in localized_patch(sql, detection)["sql"]


def test_memory_retrieval_is_family_filtered_and_schema_independent(tmp_path: Path):
    signature = build_signature("List each school", [], "SELECT COUNT(name) FROM schools", {"row_count": 1})
    rows = [{"identity": "operation-example", "error_type": "operation", "minimal_patch": "remove_unrequested_aggregate",
             "signature": signature.model_dump()},
            {"identity": "join-example", "error_type": "join", "minimal_patch": "restore_required_join",
             "signature": signature.model_dump()}]
    path = tmp_path / "memory.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = retrieve(path, signature, "operation")
    assert [row["identity"] for row in result] == ["operation-example"]
