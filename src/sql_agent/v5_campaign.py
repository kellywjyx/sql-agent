"""Answer-free V5 text-to-SQL capture; gold SQL stays in scoring callbacks."""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_evals import Metric, run_suite
from llm_evals.campaign import open_dataset
from llm_evals.capture import TimedClient

from .agent import SQLAgent
from .integrity import changed_databases, snapshot_database_hashes
from .policy import POLICY
from .retrieval_metrics import aggregate_retrieval
from .schema_retrieval import HybridSchemaRetriever
from .v3_metrics import Scoring


MODES = {
    "llama_full": {"model": "llama3.1:8b", "schema_mode": "full", "review": False},
    "coder_full": {"model": "qwen2.5-coder:7b-instruct", "schema_mode": "full", "review": False},
    "coder_hybrid": {"model": "qwen2.5-coder:7b-instruct", "schema_mode": "hybrid", "review": False},
    "coder_hybrid_review": {"model": "qwen2.5-coder:7b-instruct", "schema_mode": "hybrid", "review": True},
}
PROMPT_VERSION = "v5.1-hybrid-schema-plan"


def _pilot(cases, size=25):
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(case.metadata["db_id"], []).append(case)
    selected, rows = [], list(groups.values())
    while len(selected) < size and any(rows):
        for group in rows:
            if group and len(selected) < size:
                selected.append(group.pop(0))
    if len(selected) != size or len({case.metadata["db_id"] for case in selected}) < 5:
        raise ValueError("V5 SQL pilot requires 25 cases across at least five databases")
    return selected


def _cases(root: Path, stage: str):
    if stage == "final":
        path = root / "v5/sql-agent/data/final.jsonl"
        return open_dataset(path, purpose="final_scoring", ledger=root / "v5/exposure.jsonl",
                            selection=root / "v5/sql-agent/selection.json")
    cases = open_dataset(root / "v3/sql-agent/data/development.jsonl",
                         purpose="pilot" if stage == "pilot" else "development_selection",
                         ledger=root / "v5/exposure.jsonl")
    return _pilot(cases) if stage == "pilot" else cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "development", "final"], required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root, config = args.artifacts.resolve(), MODES[args.mode]
    cases = _cases(root, args.stage)
    before = snapshot_database_hashes(cases)
    scorer = Scoring(cases)
    client = TimedClient(model=config["model"])
    model_identity = client.check()
    retriever = HybridSchemaRetriever(root) if config["schema_mode"] == "hybrid" else None

    retrieval_metrics = [
        Metric("schema_table_recall", lambda cs, ps: aggregate_retrieval(cs, ps)["schema_table_recall"], version="5-valid-reference"),
        Metric("schema_column_recall", lambda cs, ps: aggregate_retrieval(cs, ps)["schema_column_recall"], version="5-valid-reference"),
        Metric("foreign_key_path_coverage", lambda cs, ps: aggregate_retrieval(cs, ps)["foreign_key_path_coverage"], version="5"),
    ]
    agent_cache = {}

    def predict(question, context):
        database = Path(context["database"])
        key = str(database.resolve())
        if key not in agent_cache:
            agent_cache[key] = SQLAgent(database, client, execution_timeout=10, artifacts=root,
                                        schema_retriever=retriever)
        return agent_cache[key].ask(question, revised_linking=True, correction=True,
                                    semantic_review=config["review"], schema_mode=config["schema_mode"])

    def details(_, predictions):
        retrieval = [row.get("retrieval_seconds", 0) for row in predictions]
        generation = [row.get("generation_seconds", 0) for row in predictions]
        aggregate = aggregate_retrieval(cases, predictions)
        return {"model_call_timings": client.timings, "reference_errors": scorer.errors,
                "invalid_reference_tables": aggregate["invalid_reference_tables"],
                "invalid_reference_columns": aggregate["invalid_reference_columns"],
                "database_hashes_before": before, "database_hash_mismatches": changed_databases(before),
                "retrieval_seconds_total": sum(retrieval), "generation_seconds_total": sum(generation),
                "scope": "internal BIRD training holdout; not Mini-Dev or leaderboard"}

    report = run_suite(cases, predict, [*scorer.metrics(), *retrieval_metrics],
        root / "v5/sql-agent/eval" / args.stage / args.mode,
        identity={**model_identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "candidate": args.mode, "prompt_version": PROMPT_VERSION,
                  "configuration": {"context_tokens": 8192, "max_tokens": 400, "max_attempts": 3,
                                    "schema_prompt_bytes": 10000, "schema_mode": config["schema_mode"],
                                    "semantic_review": config["review"]}},
        thresholds={"execution_accuracy_v3": {"min": 0}, "reference_coverage": {"min": .95}},
        details=details,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction["status"] != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "expected_sql": case.expected["sql"], "actual_sql": prediction.get("sql"),
            "db_id": case.metadata["db_id"],
            "attempt_categories": [attempt.get("category") or attempt.get("status") for attempt in prediction.get("attempts", [])]},
        resume=args.resume)
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
