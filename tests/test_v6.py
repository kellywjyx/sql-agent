import asyncio
import hashlib
import json
import sqlite3

import httpx

from llm_evals.local import Completion
from sql_agent.agent import SQLAgent
from sql_agent.api import create_app
from sql_agent.m_schema import render_m_schema, split_question_evidence
from sql_agent.question_plan import question_plan
from sql_agent.schema import inspect_schema
from sql_agent.semantic import enrich_plan_with_profiles, semantic_report
from sql_agent.value_profiles import ValueProfiler


class SequenceModel:
    def __init__(self, sqls):
        self.sqls = iter(sqls)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return Completion(json.dumps({"sql": next(self.sqls)}), 10, 5, "fixture")


def formatted_database(tmp_path):
    database = tmp_path / "apps.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE stores(id INTEGER PRIMARY KEY, region TEXT)")
        connection.execute("CREATE TABLE apps(id INTEGER PRIMARY KEY, store_id INTEGER REFERENCES stores(id), app TEXT, installs TEXT, price TEXT, salary TEXT, created TEXT)")
        connection.executemany("INSERT INTO stores VALUES(?, ?)", [(1, "WI"), (2, "CA")])
        connection.executemany("INSERT INTO apps VALUES(?,?,?,?,?,?,?)", [
            (1, 1, "Alpha", "1,000+", "0", "$1,200.00", "2026/08/01"),
            (2, 2, "Beta", "900+", "1.99", "$900.00", "2026/08/02"),
        ])
    return database


def test_value_profile_is_bounded_identified_and_read_only(tmp_path):
    database = formatted_database(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    profiler = ValueProfiler(tmp_path / "artifacts")
    profile = profiler.build(database, inspect_schema(database))
    installs = next(row for row in profile["columns"] if row["column"] == "installs")
    salary = next(row for row in profile["columns"] if row["column"] == "salary")
    created = next(row for row in profile["columns"] if row["column"] == "created")
    assert {"thousands_separator", "plus_suffix"} <= set(installs["format_tags"])
    assert "currency_text" in salary["format_tags"]
    assert "date_text" in created["format_tags"]
    assert profiler.load(database, inspect_schema(database))["identity"] == profile["identity"]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_m_schema_separates_evidence_and_preserves_keys_under_budget(tmp_path):
    database = formatted_database(tmp_path)
    question, evidence = split_question_evidence("top apps\nBenchmark evidence: free means price = 0")
    assert question == "top apps" and evidence == "free means price = 0"
    rendered, meta = render_m_schema(database, question, inspect_schema(database), byte_budget=650)
    assert "[Table] apps" in rendered and "FK->stores.id" in rendered
    assert meta["prompt_bytes"] <= 650 and meta["schema_format"].startswith("m-schema")


def test_semantic_checks_storage_cleanup_join_literals_and_limit(tmp_path):
    database = formatted_database(tmp_path)
    schema = inspect_schema(database)
    selection = {"selected_columns": [{"table": "apps", "column": "app", "exact_match": True}],
                 "join_paths": [], "selected_tables": ["apps"]}
    plan = question_plan("What are the top 5 installed free apps?\nBenchmark evidence: free means price = '0'", selection, [])
    plan = enrich_plan_with_profiles(plan, [{"table": "apps", "column": "installs",
        "format_tags": ["thousands_separator", "plus_suffix"], "values": ["1,000+"]}])
    bad = semantic_report("What are the top 5 installed free apps?\nBenchmark evidence: free means price = '0'",
        "SELECT app FROM apps WHERE price='0' ORDER BY installs DESC LIMIT 5", schema, plan)
    assert not bad["checks"]["numeric_cleanup"]
    good = semantic_report("What are the top 5 installed free apps?\nBenchmark evidence: free means price = '0'",
        "SELECT app FROM apps WHERE price='0' ORDER BY CAST(REPLACE(REPLACE(installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT 5",
        schema, plan)
    assert good["passed"]


def test_semantic_join_handles_implicit_foreign_key_target(tmp_path):
    database = tmp_path / "implicit.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("CREATE TABLE child(parent_id INTEGER REFERENCES parent, value TEXT)")
    schema = inspect_schema(database)
    selection = {"selected_columns": [], "join_paths": [], "selected_tables": ["parent", "child"]}
    plan = question_plan("List child values with parent names", selection, [])
    report = semantic_report("List child values with parent names",
        "SELECT child.value, parent.name FROM child JOIN parent ON child.parent_id = parent.id",
        schema, plan)
    assert report["checks"]["join_predicates"]


def test_single_character_profile_value_requires_explicit_literal():
    selection = {"selected_columns": [], "join_paths": [], "selected_tables": ["schools"]}
    hint = [{"table": "schools", "column": "virtual", "values": ["F"]}]
    unrelated = question_plan("List the names of all schools", selection, hint)
    explicit = question_plan("List schools marked virtual. Benchmark evidence: virtual = 'F'", selection, hint)
    assert unrelated["filter_columns"] == []
    assert explicit["filter_columns"] == ["schools.virtual"]


def test_adaptive_two_uses_second_candidate_only_for_mechanical_failure(tmp_path):
    database = formatted_database(tmp_path)
    artifacts = tmp_path / "artifacts"
    ValueProfiler(artifacts).build(database, inspect_schema(database))
    model = SequenceModel([
        "SELECT app FROM apps WHERE price='0' ORDER BY installs DESC LIMIT 5",
        "SELECT app FROM apps WHERE price='0' ORDER BY CAST(REPLACE(REPLACE(installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT 5",
    ])
    result = SQLAgent(database, model, artifacts=artifacts).ask(
        "What are the top 5 installed free apps?\nBenchmark evidence: free means price = '0'",
        schema_mode="full", revised_linking=True, model_profile="qwen_m_schema",
        generation_strategy="adaptive_two", value_mode="profiled")
    assert result["status"] == "completed" and model.calls == 2
    assert len(result["candidates"]) == 2 and result["semantic_checks"]["passed"], result["semantic_checks"]["hard_diagnostics"]
    assert result["value_profile_identity"] and result["schema_format"].startswith("m-schema")


def test_v6_frozen_qwen_control_cannot_generate_third_candidate(tmp_path):
    database = formatted_database(tmp_path)
    model = SequenceModel([
        "SELECT missing FROM apps",
        "SELECT still_missing FROM apps",
        "SELECT app FROM apps",
    ])
    result = SQLAgent(database, model, max_attempts=2).ask(
        "List apps", schema_mode="full", revised_linking=True,
        model_profile="qwen_v5", generation_strategy="single", value_mode="probe")
    assert result["status"] == "failed"
    assert model.calls == 2 and len(result["attempts"]) == 2


def test_api_v6_fields_are_optional_and_forwarded(tmp_path):
    class Client:
        def check(self): return {"model": "fixture"}

    class Agent:
        def __init__(self):
            self.database = tmp_path / "demo.sqlite"; self.database.write_bytes(b"fixture")
            self.client, self.schema_retriever, self.kwargs = Client(), None, None
        def ask(self, question, **kwargs):
            self.kwargs = kwargs
            return {"sql": "SELECT 1", "rows": [[1]], "columns": ["1"], "row_count": 1,
                    "attempts": [], "candidates": [], "output": [[1]], "needs_review": False}

    agent = Agent()
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(agent)), base_url="http://test") as client:
            return await client.post("/query", json={"question": "show one", "model_profile": "arctic_sql",
                "generation_strategy": "adaptive_two", "value_mode": "profiled"})
    response = asyncio.run(request())
    assert response.status_code == 200
    assert agent.kwargs["model_profile"] == "arctic_sql"
    assert agent.kwargs["generation_strategy"] == "adaptive_two"
    assert agent.kwargs["value_mode"] == "profiled"
