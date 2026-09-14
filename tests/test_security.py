import hashlib
import json
import sqlite3

import pytest

from llm_evals.local import Completion
from sql_agent.agent import SQLAgent
from sql_agent.demo import create_demo
from sql_agent.evaluation import equivalent
from sql_agent.execution import execute_sql
from sql_agent.guardrails import UnsafeSQL, validate


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_demo(path)
    return path


@pytest.mark.parametrize("sql", [
    "DELETE FROM customers", "SELECT 1; DROP TABLE customers", "PRAGMA writable_schema=ON",
    "ATTACH DATABASE 'other.db' AS extra", "SELECT load_extension('evil')", "SELECT readfile('/etc/passwd')",
    "SELECT * FROM sqlite_master", "SELECT * FROM pragma_table_info('customers')",
    "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x",
    "SELECT * FROM other.customers", "SELECT (SELECT load_extension('x'))", "INSERT INTO customers VALUES(9,'x','y')",
])
def test_unsafe_sql_rejected(sql):
    with pytest.raises(UnsafeSQL):
        validate(sql)


def test_read_only_join_aggregate_empty_and_unchanged(database):
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert execute_sql(database, "SELECT COUNT(*) FROM customers")["rows"] == [[4]]
    result = execute_sql(database, "SELECT SUM(i.quantity*i.unit_price) FROM order_items i JOIN orders o ON o.id=i.order_id WHERE o.status='completed'")
    assert result["rows"] == [[565.0]]
    assert execute_sql(database, "SELECT name FROM products WHERE price > 1000")["rows"] == []
    assert execute_sql(database, "WITH x AS (SELECT name FROM customers) SELECT * FROM x")["row_count"] == 4
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_authorizer_blocks_function_even_after_ast(database, monkeypatch):
    from sql_agent import worker
    # Even a parser regression must not bypass the independent SQLite authorizer.
    monkeypatch.setattr(worker, "validate", lambda sql: sql)
    with pytest.raises(sqlite3.DatabaseError, match="authorized"):
        worker.execute({"database": str(database), "sql": "SELECT RANDOMBLOB(999999999)"})


def test_row_budget_does_not_silently_truncate(database):
    with pytest.raises(ValueError, match="exceeds"):
        execute_sql(database, "SELECT * FROM customers", max_rows=1)


def test_timeout(database):
    with pytest.raises((TimeoutError, ValueError)):
        execute_sql(database, "SELECT SUM(a.id+b.id+c.id+d.id+e.id+f.id+g.id) FROM customers a CROSS JOIN customers b CROSS JOIN customers c CROSS JOIN customers d CROSS JOIN customers e CROSS JOIN customers f CROSS JOIN customers g", timeout=0)


def test_unknown_schema(database):
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE unusual_catalog(sku TEXT, quantity INTEGER)")
        db.execute("INSERT INTO unusual_catalog VALUES('x',7)")
    assert execute_sql(database, "SELECT SUM(quantity) FROM unusual_catalog")["rows"] == [[7]]


def test_worker_unicode_output(database):
    assert execute_sql(database, "SELECT 'café 中文' AS greeting")["rows"] == [["café 中文"]]


def test_wide_schema_separate_result_limit_and_time_function(database):
    # Synthetic regression for the frozen campaign's compatibility limit.
    # This confirms a valid wide database is not necessarily a corrupt database.
    with pytest.raises(ValueError, match="not authorized"):
        execute_sql(database, "SELECT CURRENT_TIMESTAMP")
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE wide_record(" + ",".join(f"c{i} INTEGER" for i in range(115)) + ")")
        assert connection.execute("SELECT COUNT(*) FROM wide_record").fetchone() == (0,)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert execute_sql(database, "SELECT COUNT(*) FROM wide_record")["rows"] == [[0]]
    with pytest.raises(ValueError, match="64-column"):
        execute_sql(database, "SELECT * FROM wide_record")
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


class FakeModel:
    def __init__(self, sqls):
        self.sqls = iter(sqls)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return Completion(json.dumps({"sql": next(self.sqls)}), 10, 5, "fixture")


def test_correction_and_empty_results(database):
    model = FakeModel(["SELECT nonexistent FROM customers", "SELECT name FROM customers WHERE id=999"])
    result = SQLAgent(database, model).ask("An unfamiliar request")
    assert result["status"] == "completed" and result["rows"] == []
    assert model.calls == 2


def test_tokenizer_error_is_recorded_and_corrected(database):
    model = FakeModel(["SELECT 'unterminated", "SELECT name FROM customers WHERE id=999"])
    result = SQLAgent(database, model).ask("a question")
    assert result["status"] == "completed"
    assert result["attempts"][0]["status"] == "rejected"
    assert "parse error" in result["attempts"][0]["error"]


def test_attempt_budget(database):
    model = FakeModel(["DELETE FROM customers"] * 3)
    result = SQLAgent(database, model).ask("delete customers")
    assert result["termination"] == "attempt_budget_exhausted"
    # The repeated normalized DELETE is rejected without spending a third
    # generation call because no different failure remains to correct.
    assert model.calls == 2


def test_duplicate_and_order_sensitive_scoring():
    gold = {"columns": ["x"], "rows": [[1], [1], [2]]}
    assert equivalent(gold, {"columns": ["other"], "rows": [[2], [1], [1]]})
    assert not equivalent(gold, {"columns": ["x"], "rows": [[1], [2]]})
    assert not equivalent(gold, {"columns": ["x"], "rows": [[2], [1], [1]]}, ordered=True)


def test_schema_linking_retains_multi_hop_bridge():
    from sql_agent.schema import linked_schema
    schema = [{"table": name, "ddl": name, "columns": [], "references": refs} for name, refs in
              [("bridge_b", ["target"]), ("bridge_a", ["bridge_b"]), ("start", ["bridge_a"]),
               ("target", []), ("other", []), ("spare", [])]]
    selected = linked_schema("start target other", schema)
    assert "bridge_a" in selected and "bridge_b" in selected
    assert "spare" not in selected


def test_value_probes_and_error_diagnosis(database):
    from sql_agent.schema import inspect_schema, value_hints
    from sql_agent.agent import error_category
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    schema = inspect_schema(database)
    assert schema[0]["column_details"]
    assert value_hints(database, "no quoted values", schema) == []
    value_hints(database, "find 'completed'", schema, max_probes=2)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert error_category(ValueError("no such column: x")) == "schema_mismatch"


def test_deterministic_question_coverage_checks_schema_and_operations(database):
    from sql_agent.coverage import coverage_report
    from sql_agent.schema import inspect_schema
    schema = inspect_schema(database)
    good = coverage_report("What is the total quantity for each product ordered by total?",
                           "SELECT product_id, SUM(quantity) AS total FROM order_items GROUP BY product_id ORDER BY total", schema)
    assert good["passed"] and good["checks"]["aggregation"] and good["checks"]["grouping"]
    bad = coverage_report("What is the total quantity for each product ordered by total?",
                          "SELECT product_id FROM order_items", schema)
    assert not bad["passed"] and {"aggregation", "grouping", "ordering"} <= set(bad["diagnostics"])


def test_coverage_failure_is_recorded_and_corrected(database):
    model = FakeModel(["SELECT product_id FROM order_items",
                       "SELECT product_id, SUM(quantity) AS total FROM order_items GROUP BY product_id ORDER BY total"])
    result = SQLAgent(database, model).ask("What is the total quantity for each product ordered by total?",
                                           revised_linking=True, correction=True, semantic_review=True)
    assert result["status"] == "completed" and len(result["attempts"]) == 2
    assert result["attempts"][0]["status"] == "coverage_rejected"
    assert result["coverage_checks"]["passed"] and result["policy_id"] == "sqlite-readonly-v3"


@pytest.mark.parametrize("member", ["../escape.sqlite", "/absolute.sqlite", "C:/escape.sqlite", "dir\\escape.sqlite"])
def test_dataset_archive_path_validation(tmp_path, member):
    import zipfile
    from sql_agent.benchmark_setup import extract
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        entry = zipfile.ZipInfo("placeholder")
        entry.filename = member  # Avoid ZipInfo's Windows separator normalization.
        handle.writestr(entry, b"not a database")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract(archive, tmp_path / "data")
    assert not (tmp_path / "escape.sqlite").exists()


def test_recovery_diagnostics_do_not_credit_wrong_or_unscorable_answers():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("sql_failure_analysis", Path(__file__).parents[1] / "scripts/analyze_failures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attempts = [{"status": "rejected", "category": "schema_mismatch"}, {"status": "executed"}]
    predictions = [{"id": key, "status": "completed", "attempts": attempts, "rows": []}
                   for key in ["correct", "wrong", "unscorable"]]
    failures = [{"id": key, "correct": False} for key in ["wrong", "unscorable"]]
    rows = module.classify(predictions, failures, {"unscorable": "gold deadline"})
    assert [row["category"] for row in rows] == ["correct", "execution_mismatch", "gold_unscorable"]
    assert all(row["execution_recovered"] and row["empty_execution"] for row in rows)
    assert sum(row["correct"] for row in rows) == 1
    assert module.gold_category("malformed database schema - too many columns on Match") == "schema_column_limit"
    assert module.gold_category("not authorized to use function: CURRENT_TIMESTAMP") == "function_authorizer"
    with pytest.raises(ValueError, match="Unaligned"):
        module.classify(predictions, failures, {"missing": "gold error"})
    with pytest.raises(ValueError, match="absent"):
        module.classify([{"id": "failed", "status": "failed"}], [], {})


def test_database_integrity_snapshot_detects_changes(tmp_path):
    from types import SimpleNamespace
    from sql_agent.integrity import changed_databases, snapshot_database_hashes
    database = tmp_path / "fixture.sqlite"
    database.write_bytes(b"original")
    before = snapshot_database_hashes([SimpleNamespace(context={"database": str(database)})])
    assert changed_databases(before) == []
    database.write_bytes(b"changed")
    assert changed_databases(before) == ["fixture.sqlite"]
