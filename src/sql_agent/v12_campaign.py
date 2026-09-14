"""V12 capture: frozen Qwen V5 control and value-grounded prompt arms on BIRD dev databases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_evals import run_suite
from llm_evals.campaign import open_dataset
from llm_evals.capture import TimedClient

from .agent import SQLAgent
from .integrity import changed_databases, snapshot_database_hashes
from .policy import POLICY
from .v7_evaluation import V7Scoring, structural_error_families


ARMS = {"control": "qwen_v5", "rules": "qwen_v12_rules", "notes": "qwen_v12_notes", "full": "qwen_v12"}
MODEL = "qwen2.5-coder:7b-instruct"
PROMPT_VERSION = "sql-v12-value-grounding-v1"
DECISION = "v12/sql-agent/development-decision.json"


def open_cases(root: Path, stage: str):
    if stage == "final":
        return open_dataset(root / "v6/sql-agent/data/final.jsonl", purpose="final_scoring",
                            ledger=root / "v12/exposure.jsonl", selection=root / DECISION)
    cases = open_dataset(root / "v6/sql-agent/data/development.jsonl",
                         purpose="pilot" if stage == "smoke" else "development_selection",
                         ledger=root / "v12/exposure.jsonl")
    return cases[:6] if stage == "smoke" else cases


def run(root: Path, stage: str, arm: str) -> dict:
    root = root.resolve()
    if stage not in {"smoke", "development", "final"} or arm not in ARMS:
        raise ValueError("stage must be smoke, development, or final; arm must be one of " + ", ".join(ARMS))
    if stage == "final":
        decision = json.loads((root / DECISION).read_text(encoding="utf-8"))
        if not decision.get("final_eligible") or arm not in {"control", decision.get("winner")}:
            raise RuntimeError("The locked final may run only the control and the frozen development winner")
    cases = open_cases(root, stage)
    before = snapshot_database_hashes(cases)
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    client = TimedClient(model=MODEL)
    identity = client.check()
    profile, agents = ARMS[arm], {}

    def predict(question, context):
        database = Path(context["database"])
        key = str(database.resolve())
        if key not in agents:
            agents[key] = SQLAgent(database, client, execution_timeout=10, artifacts=root)
        return agents[key].ask(question, revised_linking=True, correction=True, semantic_review=False,
                               schema_mode="full", model_profile=profile, generation_strategy="single",
                               value_mode="probe", prompt_style="direct")

    def details(_, predictions):
        return {"model_call_timings": client.timings, "database_hashes_before": before,
                "database_hash_mismatches": changed_databases(before),
                "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
                "notes_bytes": [row.get("schema_selection", {}).get("notes_bytes") for row in predictions],
                "scope": f"V12 {stage}; BIRD dev databases (V6 {'locked final' if stage == 'final' else 'development'} split)",
                "locked_final_opened": stage == "final", "api_cost_usd": 0}

    output = root / "v12/sql-agent" / ("smoke" if stage == "smoke" else "eval") / stage / arm
    # The V5 path records no candidate list, so candidate-level metrics would read a misleading 0.0.
    metrics = [metric for metric in scorer.metrics(include_ir=False)
               if metric.name not in {"first_candidate_accuracy", "candidate_oracle_ex"}]
    return run_suite(cases, predict, metrics, output,
        identity={**identity, "mode": "live", "candidate": arm, "model_profile": profile,
                  "prompt_version": PROMPT_VERSION, "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "configuration": {"schema_mode": "full", "semantic_review": False, "max_attempts": 3,
                                    "generation_strategy": "single", "value_mode": "probe"}},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "development", "final"], required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    args = parser.parse_args()
    report = run(args.artifacts, args.stage, args.arm)
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
