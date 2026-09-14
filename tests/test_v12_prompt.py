import json
import sqlite3

import pytest

from sql_agent.agent import SQLAgent
from sql_agent.generation import V12_RULES


class _Completion:
    input_tokens = output_tokens = 1

    def __init__(self, sql):
        self._sql = sql

    def json(self):
        return {"sql": self._sql}


class _Client:
    model = "qwen2.5-coder:7b-instruct"

    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append([dict(message) for message in messages])
        return _Completion("SELECT name FROM employee")


def _ask(tmp_path, profile):
    database = tmp_path / "shop" / "shop.sqlite"
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE employee (id INTEGER PRIMARY KEY, name TEXT, salary TEXT)")
        connection.execute("INSERT INTO employee (name, salary) VALUES ('Ann', 'US$57,500.00')")
    client = _Client()
    agent = SQLAgent(database, client, execution_timeout=10, artifacts=tmp_path / "artifacts")
    result = agent.ask("List employee name and salary", revised_linking=True, correction=True, semantic_review=False,
                       schema_mode="full", model_profile=profile, generation_strategy="single",
                       value_mode="probe", prompt_style="direct")
    system, user = client.calls[0][0]["content"], json.loads(client.calls[0][1]["content"])
    return result, system, user["untrusted_schema"]


@pytest.mark.parametrize("profile, rules, notes", [
    ("qwen_v5", False, False), ("qwen_v12_rules", True, False),
    ("qwen_v12_notes", False, True), ("qwen_v12", True, True),
    ("qwen_v13_notes", False, True), ("qwen_v13", True, True)])
def test_v12_arms_share_the_v5_path_and_differ_only_in_rules_and_notes(tmp_path, profile, rules, notes):
    result, system, schema = _ask(tmp_path, profile)
    assert result["status"] == "completed" and result["termination"] == "query_executed"
    assert system.startswith("Generate a single read-only SQLite SELECT query.")
    assert (V12_RULES in system) is rules
    assert ("### Column notes" in schema) is notes
    assert schema.startswith("CREATE TABLE") or "employee" in schema.split("### Column notes")[0]
