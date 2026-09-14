import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys

import httpx
import pytest

from llm_evals.local import Completion
from sql_agent.agent import SQLAgent
from sql_agent.api import create_app
from sql_agent.demo import create_demo
from sql_agent.generation import MODEL_PROFILES
from sql_agent.schema import inspect_schema
from sql_agent.semantic_ir import (FilterIntent, GroundedIntent, GroundingCandidate, MeasureIntent,
                                   OrderingIntent, ProjectionIntent, SemanticIntent, ground_intent,
                                   reference_structure)
from sql_agent.talon import gate
from sql_agent.v7_critic import (calibrate_rules, diversity, intent_critic, localized_repair,
                                 result_signals, select_candidate)
from sql_agent.v7_data import open_cases
from sql_agent.v7_evaluation import load_official_calculate_ex
from sql_agent.value_index import ValueIndex, normalize_value, question_phrases


class SequenceModel:
    def __init__(self, outputs, model="fixture"):
        self.outputs = iter(outputs)
        self.model = model
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return Completion(next(self.outputs), 10, 5, self.model, generation_seconds=.01)

    def check(self):
        return {"model": self.model, "digest": "fixture"}


def grounded_database(tmp_path):
    database = tmp_path / "grounded.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE stores(id INTEGER PRIMARY KEY, region TEXT)")
        connection.execute("CREATE TABLE apps(id INTEGER PRIMARY KEY, store_id INTEGER REFERENCES stores(id), app TEXT, price TEXT)")
        connection.executemany("INSERT INTO stores VALUES(?, ?)", [(1, "WI"), (2, "CA")])
        connection.executemany("INSERT INTO apps VALUES(?, ?, ?, ?)", [(1, 1, "Alpha", "$1,200"), (2, 2, "Beta", "900")])
    return database


def selection_for(schema):
    return {"mode": "full", "selected_tables": [row["table"] for row in schema],
        "selected_columns": [{"table": row["table"], "column": column["name"]}
                             for row in schema for column in row["column_details"]],
        "join_paths": [[row["table"], key["table"]] for row in schema for key in row["foreign_keys"]],
        "prompt_bytes": 100, "retrieval_seconds": 0}


def test_value_index_uses_whole_column_counts_and_question_conditioned_evidence(tmp_path):
    database = grounded_database(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    schema = inspect_schema(database)
    index = ValueIndex(tmp_path / "artifacts")
    manifest = index.build(database, schema)
    loaded, path = index.load(database, schema)
    assert loaded["identity"] == manifest["identity"] and path.exists()
    with sqlite3.connect(path) as connection:
        region = connection.execute("SELECT row_count, null_count, distinct_count, sqlite_types_json FROM columns WHERE table_name='stores' AND column_name='region'").fetchone()
    assert region[:3] == (2, 0, 2) and json.loads(region[3]) == {"text": 2}
    packet = index.evidence_packet(database, "List apps in region 'WI'", schema, selection_for(schema))
    assert any(row.get("printed_value") == "WI" for row in packet.facts)
    assert packet.byte_count <= 4096 and len(packet.facts) <= 12 and packet.probe_count <= 4
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_value_index_modes_and_phrase_normalization(tmp_path):
    database = grounded_database(tmp_path)
    schema = inspect_schema(database)
    index = ValueIndex(tmp_path / "artifacts")
    index.build(database, schema)
    none = index.evidence_packet(database, "List 'WI'", schema, selection_for(schema), evidence_mode="none")
    oracle = index.evidence_packet(database, "List 'WI'", schema, selection_for(schema),
                                   oracle_evidence="WI means Wisconsin", evidence_mode="oracle")
    assert none.facts == []
    assert oracle.facts[0]["kind"] == "oracle_benchmark_evidence"
    assert normalize_value(" Yes  ") == "yes"
    assert "$1,200" in question_phrases("Show values above $1,200 in New York")


def test_grounded_ir_preserves_ambiguity_and_builds_fk_skeleton(tmp_path):
    database = grounded_database(tmp_path)
    schema = inspect_schema(database)
    intent = SemanticIntent(result_grain="one row per app",
        projections=[ProjectionIntent(concept="app", role="label")],
        filters=[FilterIntent(concept="region", operator="=", value="WI")],
        join_concepts=["app", "store"])
    evidence = [{"kind": "value", "table": "stores", "column": "region", "printed_value": "WI",
                 "matched_phrase": "WI"}]
    grounded = ground_intent(intent, schema, evidence, selection_for(schema))
    assert grounded.bindings["app"][0].identifier == "apps.app"
    assert grounded.bindings["region"][0].identifier == "stores.region"
    assert grounded.join_skeletons and "JOIN" in grounded.join_skeletons[0]


def test_intent_critic_uses_exact_shape_and_result_evidence(tmp_path):
    database = grounded_database(tmp_path)
    schema = inspect_schema(database)
    intent = SemanticIntent(result_grain="one row per region",
        projections=[ProjectionIntent(concept="region", role="label")],
        measures=[MeasureIntent(concept="app", aggregation="count", distinct=True)],
        grouping=["region"], ordering=OrderingIntent(concept="app count", direction="desc"), limit=1,
        distinct_required=False)
    grounded = ground_intent(intent, schema, [], selection_for(schema))
    good_sql = "SELECT stores.region, COUNT(DISTINCT apps.app) FROM apps JOIN stores ON apps.store_id=stores.id GROUP BY stores.region ORDER BY COUNT(DISTINCT apps.app) DESC LIMIT 1"
    result = {"columns": ["region", "count"], "rows": [["WI", 1]], "row_count": 1}
    good = intent_critic(good_sql, grounded, schema, [], result)
    bad = intent_critic("SELECT COUNT(app) FROM apps", grounded, schema, [], result)
    assert good["passed"], good["actionable_diagnostics"]
    assert {"projection_count", "ordering", "limit"} & set(bad["actionable_diagnostics"])
    assert good["result_signals"]["row_count"] == 1


def test_critic_calibration_and_localized_repair(tmp_path):
    reports = [
        {"checks": {"projection_count": False, "numeric_cleanup": True}},
        {"checks": {"projection_count": False, "numeric_cleanup": False}},
        {"checks": {"projection_count": True, "numeric_cleanup": False}},
    ]
    calibration = calibrate_rules(reports, [False, False, True], minimum_precision=.9)
    assert "projection_count" in calibration["allowed_rules"]
    assert "numeric_cleanup" not in calibration["allowed_rules"]
    intent = SemanticIntent(result_grain="one row", projections=[ProjectionIntent(concept="app")], limit=1)
    grounded = GroundedIntent(semantic_intent=intent)
    repaired = localized_repair("SELECT app FROM apps", "limit", grounded, [])
    assert repaired and repaired["changed_dimensions"] == ["limit"] and "LIMIT 1" in repaired["sql"]


def test_candidate_diversity_result_voting_and_uncertainty():
    first_result = {"columns": ["x"], "rows": [[1], [2]], "row_count": 2}
    second_result = {"columns": ["x"], "rows": [[2], [1]], "row_count": 2}
    third_result = {"columns": ["x"], "rows": [[3]], "row_count": 1}
    candidates = []
    for index, result in enumerate([first_result, second_result, third_result]):
        candidates.append({"path": str(index), "status": "executed", "sql": f"SELECT {index}", "result": result,
            "result_signals": result_signals(result), "critic": {"actionable_diagnostics": [], "risk": index / 10}})
    selected, decision = select_candidate(candidates)
    assert selected["path"] in {"0", "1"}
    assert decision["reason"] == "majority_result_then_intent_critic" and not decision["needs_review"]
    assert diversity("SELECT app FROM apps", "SELECT DISTINCT app FROM apps")["dimensions"] == ["distinct"]


def test_result_vote_marks_disagreement_for_review():
    candidates = []
    for index in range(2):
        result = {"columns": ["x"], "rows": [[index]], "row_count": 1}
        candidates.append({"path": str(index), "status": "executed", "sql": f"SELECT {index}",
            "result": result, "result_signals": result_signals(result),
            "critic": {"actionable_diagnostics": [], "risk": 0}})
    _, decision = select_candidate(candidates)
    assert decision["needs_review"] and decision["reason"] == "intent_critic_no_result_majority"


def test_official_scorer_is_checksum_verified_and_executed_in_isolation(tmp_path):
    scorer = tmp_path / "evaluation_ex.py"
    scorer.write_text("def calculate_ex(predicted_res, ground_truth_res):\n"
                      "    return int(set(predicted_res) == set(ground_truth_res))\n", encoding="utf-8")
    digest = hashlib.sha256(scorer.read_bytes()).hexdigest()
    calculate = load_official_calculate_ex(scorer, digest)
    assert calculate([(1,), (2,)], [(2,), (1,)]) == 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_official_calculate_ex(scorer, "0" * 64)


def test_reference_structure_covers_grain_and_set_operations():
    value = reference_structure("SELECT region, COUNT(*) FROM stores GROUP BY region ORDER BY COUNT(*) DESC LIMIT 1")
    assert value["projection_count"] == 2 and value["grouping"] and value["ordering"] and value["limit"]
    assert reference_structure("SELECT region FROM stores UNION SELECT app FROM apps")["set_operation"] == "union"


def test_v7_agent_runs_two_paths_and_hides_scoring_candidate_results(tmp_path):
    database = grounded_database(tmp_path)
    artifacts = tmp_path / "artifacts"
    ValueIndex(artifacts).build(database, inspect_schema(database))
    intent = SemanticIntent(result_grain="one row per app",
        projections=[ProjectionIntent(concept="app", role="label")],
        filters=[FilterIntent(concept="region", operator="=", value="WI")], join_concepts=["store"])
    qwen = SequenceModel([intent.model_dump_json()], model="qwen2.5-coder:7b-instruct")
    arctic = SequenceModel([
        "SELECT apps.app FROM apps JOIN stores ON apps.store_id=stores.id WHERE stores.region='WI'",
        "SELECT app FROM stores JOIN apps ON stores.id=apps.store_id WHERE region='WI'",
    ], model="arctic")
    agent = SQLAgent(database, qwen, artifacts=artifacts, clients={"arctic_sql": arctic, "qwen_v5": qwen})
    result = agent.ask("List app in region 'WI'", pipeline_profile="v7_grounded", schema_mode="auto",
                       generation_strategy="adaptive_two", candidate_paths=["arctic_direct", "arctic_ir"])
    assert result["status"] == "completed" and result["pipeline_profile"] == "v7_grounded"
    assert result["rows"] == [["Alpha"]] and len(result["candidates"]) == 2
    assert all("result" not in candidate for candidate in result["candidates"])
    assert result["candidate_result_group"]["reason"] == "majority_result_then_intent_critic"


def test_v7_evaluation_capture_can_retain_candidate_results(tmp_path):
    database = grounded_database(tmp_path)
    artifacts = tmp_path / "artifacts"
    ValueIndex(artifacts).build(database, inspect_schema(database))
    intent = SemanticIntent(projections=[ProjectionIntent(concept="app")])
    qwen = SequenceModel([intent.model_dump_json()], model="qwen2.5-coder:7b-instruct")
    arctic = SequenceModel(["SELECT app FROM apps"], model="arctic")
    agent = SQLAgent(database, qwen, artifacts=artifacts, clients={"arctic_sql": arctic, "qwen_v5": qwen})
    result = agent.ask("List app", pipeline_profile="v7_grounded", candidate_paths=["arctic_direct"],
                       capture_candidate_results=True)
    assert result["candidates"][0]["result"]["rows"] == [["Alpha"], ["Beta"]]


def test_api_exposes_v7_profile_but_keeps_v5_single_default(tmp_path):
    class Agent:
        def __init__(self):
            self.database = tmp_path / "demo.sqlite"; self.database.write_bytes(b"fixture")
            self.client = SequenceModel([]); self.schema_retriever = None; self.kwargs = None
        def ask(self, question, **kwargs):
            self.kwargs = kwargs
            return {"sql": "SELECT 1", "rows": [[1]], "columns": ["1"], "row_count": 1,
                    "attempts": [], "output": [[1]]}
    agent = Agent()
    async def request(payload):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(agent)), base_url="http://test") as client:
            return await client.post("/query", json=payload)
    response = asyncio.run(request({"question": "show one"}))
    assert response.status_code == 200
    assert agent.kwargs["pipeline_profile"] == "v5" and agent.kwargs["generation_strategy"] == "single"
    response = asyncio.run(request({"question": "show one", "pipeline_profile": "v7_grounded"}))
    assert response.status_code == 200 and agent.kwargs["pipeline_profile"] == "v7_grounded"


def test_locked_final_is_not_a_v7_openable_role(tmp_path):
    with pytest.raises(ValueError, match="never opens"):
        open_cases(tmp_path, "final")


def test_talon_gate_remains_closed_without_selection_or_approved_asset(tmp_path):
    value = gate(tmp_path)
    assert not value["eligible"] and "selection" in value["reasons"][0].casefold()


def test_q5_profile_is_evaluation_only_and_public_api_rejects_it(tmp_path):
    assert MODEL_PROFILES["arctic_sql_q5"]["model"].endswith(":Q5_K_M")
    class Agent:
        database = tmp_path / "demo.sqlite"
        client = SequenceModel([])
        schema_retriever = None
    Agent.database.write_bytes(b"fixture")
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(Agent())),
                                     base_url="http://test") as client:
            return await client.post("/query", json={"question": "show rows", "model_profile": "arctic_sql_q5"})
    response = asyncio.run(request())
    assert response.status_code == 422


def test_q5_campaign_rejects_any_size_other_than_twenty(tmp_path):
    script = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts/run_sql_v7_eval.py"
    process = subprocess.run([sys.executable, str(script), "--artifacts", str(tmp_path),
        "--stage", "reproduction", "--mode", "reference_q5_oracle", "--pilot-size", "19"],
        capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert "restricted to exactly --pilot-size 20" in process.stderr
