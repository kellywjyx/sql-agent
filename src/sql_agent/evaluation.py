import json
from collections import Counter
from pathlib import Path

from llm_evals import EvalCase

from .execution import execute_sql


def equivalent(gold: dict, predicted: dict, ordered: bool = False) -> bool:
    """Local EX: duplicate-preserving rows, positional columns, explicit ordering."""
    if len(gold["columns"]) != len(predicted["columns"]):
        return False
    def encode(rows):
        return [tuple(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for v in row) for row in rows]
    expected, actual = encode(gold["rows"]), encode(predicted["rows"])
    return expected == actual if ordered else Counter(expected) == Counter(actual)


def execution_metric(cases, predictions):
    correct = 0
    for case, prediction in zip(cases, predictions):
        if prediction.get("status") != "completed":
            continue
        gold = execute_sql(Path(case.context["database"]), case.expected["sql"])
        correct += equivalent(gold, prediction, case.expected.get("ordered", False))
    return correct / len(cases)


def load_benchmark(path: Path, database_root: Path, kind: str) -> list[EvalCase]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for i, record in enumerate(records):
        identifier = record["db_id"]
        database = (database_root / identifier / f"{identifier}.sqlite").resolve()
        if not database.is_relative_to(database_root.resolve()) or not database.is_file():
            raise ValueError(f"Missing or invalid benchmark database: {identifier}")
        sql = record.get("SQL") or record.get("query")
        question = record["question"]
        if record.get("evidence"):
            question += "\nBenchmark evidence: " + record["evidence"]
        cases.append(EvalCase(id=f"{kind}-{i}", input=question, expected={"sql": sql, "ordered": "order by" in sql.lower()},
                              context={"database": str(database)}, metadata={"benchmark": kind,
                              "difficulty": record.get("difficulty", "unavailable"), "db_id": identifier,
                              "upstream_question_id": record.get("question_id", i)}))
    return cases
