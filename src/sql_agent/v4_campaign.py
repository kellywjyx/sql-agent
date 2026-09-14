"""Local V4 text-to-SQL model comparison on the frozen V3 cases."""
from pathlib import Path
from llm_evals import run_suite
from llm_evals.capture import campaign_cases, command_arguments, TimedClient
from .agent import SQLAgent
from .policy import POLICY
from .v3_metrics import Scoring
from .integrity import changed_databases, snapshot_database_hashes


def main():
    args = command_arguments(["llama_control", "coder_candidate"])
    root = args.artifacts.resolve()
    cases = campaign_cases(root, "sql-agent", args.stage, pilot_size=25, run_version="v4")
    database_hashes_before = snapshot_database_hashes(cases)
    scorer = Scoring(cases)
    model = args.model or ("qwen2.5-coder:7b-instruct" if args.mode == "coder_candidate" else "llama3.1:8b")
    client = TimedClient(model=model)
    identity = client.check()
    report = run_suite(cases, lambda value, context: SQLAgent(
        Path(context["database"]), client, execution_timeout=10).ask(
            value, revised_linking=True, correction=True, semantic_review=True),
        scorer.metrics(), root / "v4/sql-agent/eval" / args.stage / args.mode,
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "candidate": args.mode, "prompt_version": "v4-schema-coverage",
                  "configuration": {"context_tokens": 8192, "max_tokens": 400, "max_attempts": 3,
                                    "schema_prompt_bytes": 10000}},
        thresholds={"execution_accuracy_v3": {"min": 0}, "reference_coverage": {"min": .95}},
        details=lambda: {"model_call_timings": client.timings, "reference_errors": scorer.errors,
                         "database_hashes_before": database_hashes_before,
                         "database_hash_mismatches": changed_databases(database_hashes_before),
                         "scope": "internal BIRD training holdout; not Mini-Dev or leaderboard"},
        analyze=lambda c, p: {"category": "reference_failure" if c.id in scorer.errors else
            "application_failure" if p["status"] != "completed" else "correct" if scorer.correct(c, p) else "wrong_result",
            "expected_sql": c.expected["sql"], "actual_sql": p.get("sql"), "db_id": c.metadata["db_id"]},
        resume=args.resume)
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
