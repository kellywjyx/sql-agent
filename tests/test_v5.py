import hashlib
import asyncio
import json
import sqlite3

import httpx
import numpy as np
import pytest

from llm_evals.local import Completion
from llm_evals import EvalCase
from sql_agent.agent import SQLAgent
from sql_agent.coverage import coverage_report
from sql_agent.demo import create_demo
from sql_agent.question_plan import question_plan
from sql_agent.schema import inspect_schema, value_hints
from sql_agent.schema_retrieval import HybridSchemaRetriever
from sql_agent.integrity import changed_databases, snapshot_database_hashes
from sql_agent.retrieval_metrics import aggregate_retrieval
from sql_agent.api import create_app


def semantic_fixture(values):
    dimensions = ["buyer person customer", "money currency amount revenue", "product item", "status state", "date year"]
    output = []
    for value in values:
        words = value.casefold()
        vector = [sum(token in words for token in group.split()) for group in dimensions]
        vector.append(0.1)
        output.append(vector)
    return np.asarray(output, dtype=np.float32)


def wide_database(tmp_path):
    database = tmp_path / "catalog.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, customer_name TEXT, currency_code TEXT)")
        connection.execute("CREATE TABLE purchases(purchase_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), amount REAL)")
        connection.execute("CREATE TABLE filler(" + ",".join(f"field_{index} TEXT" for index in range(45)) + ")")
    descriptions = tmp_path / "database_description"
    descriptions.mkdir()
    (descriptions / "customers.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "customer_name,Customer Name,person who buys products,text,\n"
        "currency_code,Currency Code,money currency used by the buyer,text,\n", encoding="utf-8")
    return database


def test_hybrid_retrieval_uses_descriptions_fk_closure_and_cache(tmp_path):
    database = wide_database(tmp_path)
    artifacts = tmp_path / "artifacts"
    retriever = HybridSchemaRetriever(artifacts, encoder=semantic_fixture)
    schema = inspect_schema(database)
    manifest = retriever.build(database, schema)
    prompt, selection = retriever.select(database, "Which buyer and money currency made each purchase?", schema)
    chosen = {(row["table"], row["column"]) for row in selection["selected_columns"]}
    assert ("customers", "customer_name") in chosen
    assert ("customers", "currency_code") in chosen
    assert {"customers", "purchases"} <= set(selection["selected_tables"])
    assert "customers.customer_id" in selection["retained_keys"]
    assert len(prompt.encode()) <= 10000
    assert retriever.build(database, schema)["schema_identity"] == manifest["schema_identity"]
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE filler ADD COLUMN changed TEXT")
    assert retriever.build(database, inspect_schema(database))["schema_identity"] != manifest["schema_identity"]


def test_missing_schema_index_has_actionable_error(tmp_path):
    database = wide_database(tmp_path)
    retriever = HybridSchemaRetriever(tmp_path / "artifacts", encoder=semantic_fixture)
    with pytest.raises(RuntimeError, match="index-schema"):
        retriever.select(database, "buyer", inspect_schema(database))


def test_value_probes_only_selected_columns_and_preserve_database(tmp_path):
    database = tmp_path / "demo.sqlite"
    create_demo(database)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    hints = value_hints(database, "orders with 'completed' status", inspect_schema(database),
                        selected_columns=[{"table": "orders", "column": "status"}])
    assert hints == [{"table": "orders", "column": "status", "values": ["completed"]}]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_question_plan_and_coverage_detect_semantic_omissions(tmp_path):
    database = tmp_path / "demo.sqlite"
    create_demo(database)
    schema = inspect_schema(database)
    selection = {"selected_tables": ["products", "order_items"], "join_paths": [["products", "order_items"]],
                 "selected_columns": [
                     {"table": "products", "column": "name", "exact_match": True},
                     {"table": "order_items", "column": "quantity", "exact_match": True}]}
    plan = question_plan("Which product name has the highest total quantity?", selection, [])
    bad = coverage_report("Which product name has the highest total quantity?",
                          "SELECT SUM(quantity) FROM order_items", schema, plan)
    assert not bad["passed"] and {"requested_projection", "ordering", "limit"} <= set(bad["diagnostics"])
    good = coverage_report("Which product name has the highest total quantity?",
        "SELECT p.name, SUM(i.quantity) AS total FROM products p JOIN order_items i ON p.id=i.product_id GROUP BY p.name ORDER BY total DESC LIMIT 1",
        schema, plan)
    assert good["passed"]


def test_superlative_order_limit_is_not_forced_into_aggregate_filter(tmp_path):
    database = tmp_path / "demo.sqlite"
    create_demo(database)
    schema = inspect_schema(database)
    selection = {"selected_tables": ["products"], "join_paths": [], "selected_columns": [
        {"table": "products", "column": "name", "exact_match": True},
        {"table": "products", "column": "price", "exact_match": True}]}
    question = "Which product name has the highest price?\nBenchmark evidence: highest refers to MAX(price)"
    plan = question_plan(question, selection, [])
    report = coverage_report(question, "SELECT name FROM products ORDER BY price DESC LIMIT 1", schema, plan)
    assert report["checks"]["aggregation"]
    assert report["checks"]["filtering"]


class SequenceModel:
    def __init__(self, sqls):
        self.sqls = iter(sqls)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return Completion(json.dumps({"sql": next(self.sqls)}), 10, 5, "fixture")


def test_duplicate_sql_attempt_is_recorded_without_reexecution(tmp_path):
    database = tmp_path / "demo.sqlite"
    create_demo(database)
    repeated = "SELECT product_id FROM order_items"
    model = SequenceModel([repeated, repeated,
        "SELECT product_id, SUM(quantity) AS total FROM order_items GROUP BY product_id ORDER BY total"])
    result = SQLAgent(database, model).ask("What is the total quantity for each product ordered by total?",
                                           schema_mode="full", revised_linking=True,
                                           correction=True, semantic_review=True)
    assert result["status"] == "failed" and len(result["attempts"]) == 2
    assert model.calls == 2
    assert result["attempts"][1]["category"] == "duplicate_candidate"
    assert result["attempts"][1]["coverage_diagnostics"] == ["duplicate_candidate"]


def test_third_call_requires_a_different_remaining_failure(tmp_path):
    database = tmp_path / "demo.sqlite"
    create_demo(database)
    model = SequenceModel([
        "SELECT product_id FROM order_items",
        "SELECT product_id FROM order_items WHERE product_id > 0",
        "SELECT product_id, SUM(quantity) FROM order_items GROUP BY product_id ORDER BY SUM(quantity)",
    ])
    result = SQLAgent(database, model).ask("What is the total quantity for each product ordered by total?",
                                           schema_mode="full", revised_linking=True,
                                           correction=True, semantic_review=True)
    assert model.calls == 2
    assert result["termination"] == "coverage_unresolved"


def test_foreign_key_connectivity_rejects_unrelated_join():
    schema = [
        {"table": "a", "column_details": [{"name": "id"}], "references": [], "foreign_keys": []},
        {"table": "b", "column_details": [{"name": "id"}], "references": [], "foreign_keys": []},
    ]
    plan = {"join_required": True, "required_output_columns": [], "filter_columns": []}
    report = coverage_report("list joined records", "SELECT a.id FROM a JOIN b ON a.id=b.id", schema, plan)
    assert not report["checks"]["foreign_key_connectivity"]


def test_retrieval_scoring_separates_invalid_gold_columns(tmp_path):
    database = tmp_path / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT)")
    case = EvalCase(id="bad-reference", input="total salary", expected={"sql": "SELECT SUM(salary) FROM employees"},
                    context={"database": str(database)}, metadata={})
    prediction = {"schema_selection": {"selected_tables": ["employees"],
        "selected_columns": [{"table": "employees", "column": "id"}, {"table": "employees", "column": "name"}],
        "retained_keys": ["employees.id"]}}
    scores = aggregate_retrieval([case], [prediction])
    assert scores["schema_table_recall"] == 1
    assert scores["schema_column_recall"] == 0
    assert scores["invalid_reference_columns"] == 1


def test_integrity_gate_uses_only_pre_post_database_bytes(tmp_path):
    database = tmp_path / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE values_table(value INTEGER)")
    case = EvalCase(id="one", input="q", expected={"sql": "SELECT value FROM values_table"},
                    context={"database": str(database)}, metadata={})
    before = snapshot_database_hashes([case])
    assert changed_databases(before) == []
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO values_table VALUES (1)")
    assert changed_databases(before) == [database.name]


def test_api_keeps_legacy_request_compatible_and_defaults_to_evaluated_v5_auto_schema(tmp_path):
    class Client:
        def check(self):
            return {"model": "fixture"}

    class Agent:
        def __init__(self):
            self.database = tmp_path / "demo.sqlite"
            self.database.write_bytes(b"fixture")
            self.client, self.schema_retriever, self.kwargs = Client(), None, None

        def ask(self, question, **kwargs):
            self.kwargs = kwargs
            return {"sql": "SELECT 1", "rows": [[1]], "columns": ["1"], "row_count": 1,
                    "attempts": [], "output": [[1]]}

    agent = Agent()
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(agent)),
                                     base_url="http://test") as client:
            return await client.post("/query", json={"question": "show one"})
    response = asyncio.run(request())
    assert response.status_code == 200
    assert agent.kwargs["schema_mode"] == "auto"
    assert agent.kwargs["pipeline_profile"] == "v5"
    assert agent.kwargs["generation_strategy"] == "single"
    assert agent.kwargs["semantic_review"] is False
