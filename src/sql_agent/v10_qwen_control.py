"""Same-set frozen Qwen V5 control for V10 promotion comparison."""
from __future__ import annotations

from pathlib import Path

from llm_evals import run_suite
from llm_evals.capture import TimedClient

from .agent import SQLAgent
from .integrity import changed_databases, snapshot_database_hashes
from .policy import POLICY
from .v7_evaluation import V7Scoring, structural_error_families
from .v10_data import open_pilot


def main():
    root = Path(__file__).resolve().parents[3] / "artifacts"
    cases = open_pilot(root); before = snapshot_database_hashes(cases)
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    client = TimedClient(model="qwen2.5-coder:7b-instruct"); agents = {}

    def predict(question, context):
        database = Path(context["database"]); key = str(database.resolve())
        if key not in agents:
            agents[key] = SQLAgent(database, client, execution_timeout=10, artifacts=root)
        return agents[key].ask(question, revised_linking=True, correction=True,
                               semantic_review=False, schema_mode="full", model_profile="qwen_v5",
                               generation_strategy="single", value_mode="probe", prompt_style="direct")

    def details(used_cases, predictions):
        return {"model_call_timings": client.timings, "database_hashes_before": before,
                "database_hash_mismatches": changed_databases(before),
                "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
                "scope": "same 40-case V10 exposed pilot; frozen Qwen V5 full-schema control",
                "api_cost_usd": 0, "locked_final_opened": False}

    identity = client.check()
    report = run_suite(cases, predict, scorer.metrics(include_ir=False), root / "v10/sql-agent/eval/pilot/qwen_v5_control",
        identity={**identity, "mode": "live", "candidate": "qwen_v5_same_set_control",
                  "prompt_version": "v5.1-hybrid-schema-plan", "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "configuration": {"schema_mode": "full", "semantic_review": False,
                                    "max_attempts": 3, "context_tokens": 8192}, "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"]})
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
