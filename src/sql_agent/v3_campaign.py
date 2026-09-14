from pathlib import Path
from llm_evals import run_suite
from llm_evals.capture import campaign_cases, command_arguments, TimedClient
from .agent import SQLAgent
from .policy import POLICY
from .v3_metrics import Scoring


def main():
    args = command_arguments(["revised_linking", "reviewed"])
    root = args.artifacts.resolve()
    cases = campaign_cases(root, "sql-agent", args.stage)
    scorer = Scoring(cases)
    client = TimedClient()
    identity = client.check()
    report = run_suite(cases, lambda value, context: SQLAgent(Path(context["database"]), client, execution_timeout=10).ask(
        value, revised_linking=True, correction=args.mode == "reviewed", semantic_review=args.mode == "reviewed"),
        scorer.metrics(), root / "v3/sql-agent/eval" / args.stage / args.mode,
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10}, "candidate": args.mode,
                  "prompt_version": "v3-schema-checklist", "configuration": {"context_tokens": 8192, "max_tokens": 400,
                  "max_attempts": 3 if args.mode == "reviewed" else 1, "schema_prompt_bytes": 10000}},
        thresholds={"execution_accuracy_v3": {"min": 0}, "reference_coverage": {"min": 0}},
        details=lambda: {"model_call_timings": client.timings, "reference_errors": scorer.errors,
                         "scope": "internal BIRD training holdout; not Mini-Dev or leaderboard"},
        analyze=lambda c, p: {"category": "reference_failure" if c.id in scorer.errors else
            "application_failure" if p["status"] != "completed" else "correct" if scorer.correct(c, p) else "wrong_result",
            "expected_sql": c.expected["sql"], "actual_sql": p.get("sql"), "db_id": c.metadata["db_id"]})
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
