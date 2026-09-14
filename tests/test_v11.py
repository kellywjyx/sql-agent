import json
from pathlib import Path

import pytest

from sql_agent.agent import SQLAgent
from sql_agent.api import QueryRequest
from sql_agent.v11_assets import verify
from sql_agent.v11_data import DB_COUNTS, SHAPE_COUNTS, _assign_databases, _token_count, query_shape
from sql_agent.v11_prompt import messages
from sql_agent.v11_training import preflight


def test_v11_query_shapes_are_mutually_exclusive():
    assert query_shape("SELECT a FROM t UNION SELECT a FROM u") == "set_or_subquery"
    assert query_shape("SELECT a FROM t JOIN u ON t.id = u.id") == "join"
    assert query_shape("SELECT category, COUNT(*) FROM t GROUP BY category") == "aggregate"
    assert query_shape("SELECT a FROM t ORDER BY a LIMIT 3") == "ordering"
    assert query_shape("SELECT a FROM t WHERE b = 1") == "simple"


def test_database_assignment_is_deterministic_and_disjoint():
    rows = [{"db_id": f"db-{index}"} for index in range(58)]
    first = _assign_databases(rows); second = _assign_databases(list(reversed(rows)))
    assert first == second
    assert {role: list(first.values()).count(role) for role in DB_COUNTS} == DB_COUNTS
    assert {role: sum(SHAPE_COUNTS[role].values()) for role in SHAPE_COUNTS} == {
        "train": 1200, "validation": 150, "development": 200}


def test_v11_prompt_has_scoped_generated_evidence_but_no_oracle_label():
    value = messages("Which private schools are open?", "CREATE TABLE schools (...);",
                     "[E1] allowed=predicate; private maps to schools.type = 'PRI'")
    rendered = json.dumps(value)
    assert "allowed=predicate" in rendered
    assert "Benchmark evidence" not in rendered
    assert "Return SQL only" in rendered


def test_token_limit_is_measured_without_truncation():
    class Tokenizer:
        def apply_chat_template(self, value, **kwargs):
            return list(range(sum(len(row["content"]) for row in value)))
    value = [{"role": "user", "content": "abc"}]
    assert _token_count(Tokenizer(), value, "SELECT 1") == len("abcSELECT 1")


def test_missing_v11_asset_is_actionable(tmp_path: Path):
    with pytest.raises(RuntimeError, match="15.2 GB download"):
        verify(tmp_path)


def test_v11_profile_requires_configured_adapter(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SQL_V11_ADAPTER", raising=False)
    agent = SQLAgent(tmp_path / "demo.sqlite", artifacts=tmp_path)
    with pytest.raises(RuntimeError, match="SQL_V11_ADAPTER"):
        agent.ask("List all rows", pipeline_profile="v11_adapter")


def test_api_accepts_experimental_v11_profile_without_changing_default():
    assert QueryRequest(question="List all rows").pipeline_profile == "v5"
    assert QueryRequest(question="List all rows", pipeline_profile="v11_adapter").pipeline_profile == "v11_adapter"


def test_preflight_attempt_identifiers_are_safe_and_immutable(tmp_path: Path):
    with pytest.raises(ValueError, match="safe lowercase identifier"):
        preflight(tmp_path, attempt_id="../overwrite")
