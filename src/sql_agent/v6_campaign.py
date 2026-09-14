"""V6 local text-to-SQL capture; references stay in scoring callbacks."""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_evals import Metric, run_suite
from llm_evals.campaign import open_dataset
from llm_evals.capture import TimedClient

from .agent import SQLAgent
from .integrity import changed_databases, snapshot_database_hashes
from .policy import POLICY
from .v3_metrics import Scoring
from .value_profiles import ValueProfiler


MODES = {
    "qwen_v5": {"profile": "qwen_v5", "strategy": "single", "value": "probe", "prompt": "direct", "candidate_limit": 2},
    "qwen_mschema": {"profile": "qwen_m_schema", "strategy": "single", "value": "profiled", "prompt": "direct", "candidate_limit": 2},
    "arctic_direct": {"profile": "arctic_sql", "strategy": "single", "value": "profiled", "prompt": "direct", "candidate_limit": 2},
    "arctic_plan": {"profile": "arctic_sql", "strategy": "single", "value": "profiled", "prompt": "plan_first", "candidate_limit": 2},
    "arctic_adaptive": {"profile": "arctic_sql", "strategy": "adaptive_two", "value": "profiled", "prompt": "direct", "candidate_limit": 2},
    "xiyan_direct": {"profile": "xiyan_sql", "strategy": "single", "value": "profiled", "prompt": "direct", "candidate_limit": 2},
    "xiyan_adaptive": {"profile": "xiyan_sql", "strategy": "adaptive_two", "value": "profiled", "prompt": "direct", "candidate_limit": 2},
}


def _pilot(cases, size=30):
    by_db = {}
    for case in cases:
        by_db.setdefault(case.metadata["db_id"], []).append(case)
    selected, rows = [], [by_db[key] for key in sorted(by_db)]
    while len(selected) < size and any(rows):
        for row in rows:
            if row and len(selected) < size:
                selected.append(row.pop(0))
    if len(selected) != size or len({case.metadata["db_id"] for case in selected}) < 10:
        raise ValueError("V6 pilot requires 30 cases across at least ten databases")
    return selected


def _cases(root: Path, stage: str):
    if stage == "final":
        selection = root / "v6/sql-agent/selection-v2.json"
        if not selection.exists():
            selection = root / "v6/sql-agent/selection.json"
        return open_dataset(root / "v6/sql-agent/data/final.jsonl", purpose="final_scoring",
            ledger=root / "v6/exposure.jsonl", selection=selection)
    cases = open_dataset(root / "v6/sql-agent/data/development.jsonl",
        purpose="pilot" if stage == "pilot" else "development_selection", ledger=root / "v6/exposure.jsonl")
    return _pilot(cases) if stage == "pilot" else cases


def build_profiles(root: Path, cases) -> list[dict]:
    from .schema import inspect_schema
    profiler, built, seen = ValueProfiler(root), [], set()
    for case in cases:
        database = Path(case.context["database"])
        if database.resolve() in seen:
            continue
        seen.add(database.resolve())
        profile = profiler.build(database, inspect_schema(database))
        built.append({"database": database.name, "identity": profile["identity"]})
    return built


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "development", "final"], required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-profiles", action="store_true")
    args = parser.parse_args()
    root, config = args.artifacts.resolve(), MODES[args.mode]
    cases = _cases(root, args.stage)
    if args.prepare_profiles:
        print(build_profiles(root, cases), flush=True)
        return
    before = snapshot_database_hashes(cases)
    scorer = Scoring(cases)
    model_name = __import__("sql_agent.generation", fromlist=["MODEL_PROFILES"]).MODEL_PROFILES[config["profile"]]["model"]
    client = TimedClient(model=model_name)
    identity = client.check()
    agent_cache = {}

    def predict(question, context):
        database = Path(context["database"])
        key = str(database.resolve())
        if key not in agent_cache:
            agent_cache[key] = SQLAgent(database, client, max_attempts=config["candidate_limit"], execution_timeout=10, artifacts=root,
                                        clients={config["profile"]: client})
        return agent_cache[key].ask(question, schema_mode="full", revised_linking=True, correction=True,
            model_profile=config["profile"], generation_strategy=config["strategy"],
            value_mode=config["value"], prompt_style=config["prompt"])

    def details(_, predictions):
        return {"model_call_timings": client.timings, "reference_errors": scorer.errors,
            "database_hashes_before": before, "database_hash_mismatches": changed_databases(before),
            "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
            "needs_review_count": sum(bool(row.get("needs_review")) for row in predictions),
            "scope": "BIRD development partition; not an official leaderboard submission"}

    report = run_suite(cases, predict, scorer.metrics(), root / "v6/sql-agent/eval" / args.stage / args.mode,
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
            "candidate": args.mode, "prompt_version": "sql-v6-specialized-v1", "configuration": config},
        thresholds={"execution_accuracy_v3": {"min": 0}, "reference_coverage": {"min": .95}},
        details=details, analyze=lambda case, prediction: {
            "category": "reference_failure" if case.id in scorer.errors else
                "application_failure" if prediction["status"] != "completed" else
                "correct" if scorer.correct(case, prediction) else "wrong_result",
            "expected_sql": case.expected["sql"], "actual_sql": prediction.get("sql"),
            "db_id": case.metadata["db_id"], "semantic_diagnostics":
                (prediction.get("semantic_checks") or {}).get("hard_diagnostics", []),
            "attempt_categories": [attempt.get("category") or attempt.get("status")
                                   for attempt in prediction.get("attempts", [])]}, resume=args.resume)
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
