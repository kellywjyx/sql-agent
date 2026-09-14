"""Build CPU indexes and score answer-free schema selection on development."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from llm_evals import load_cases
from llm_evals.campaign import immutable_json

from .retrieval_metrics import aggregate_retrieval
from .schema import inspect_schema
from .schema_retrieval import HybridSchemaRetriever


def evaluate(root: Path, dataset: Path, label: str):
    cases = load_cases(dataset)
    retriever = HybridSchemaRetriever(root)
    schemas = {}
    indexes = {}
    for case in cases:
        database = Path(case.context["database"])
        key = str(database.resolve())
        if key not in schemas:
            schemas[key] = inspect_schema(database)
            indexes[key] = retriever.build(database, schemas[key])
    predictions = []
    for case in cases:
        database = Path(case.context["database"])
        _, selection = retriever.select(database, case.input, schemas[str(database.resolve())])
        predictions.append({"id": case.id, "schema_selection": selection})
    metrics = aggregate_retrieval(cases, predictions)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    target = root / "v5/sql-agent/retrieval" / label / run_id
    immutable_json(target / "report.json", {"status": "passed" if metrics["schema_table_recall"] >= .98 and metrics["schema_column_recall"] >= .95 else "failed",
        "dataset": str(dataset), "count": len(cases), "databases": len(schemas), "metrics": metrics,
        "thresholds": {"schema_table_recall": .98, "schema_column_recall": .95},
        "indexes": {Path(path).name: value["schema_identity"] for path, value in indexes.items()},
        "api_cost_usd": 0, "gold_visibility": "required tables/columns used only after application retrieval"})
    with (target / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target / "report.json"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    print(evaluate(args.artifacts.resolve(), args.dataset.resolve(), args.label))


if __name__ == "__main__":
    main()
