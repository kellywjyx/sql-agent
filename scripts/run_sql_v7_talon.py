"""Run the approval-gated Talon critic on exposed V7 calibration material."""
import argparse
import json
from pathlib import Path

from llm_evals import Metric, run_suite

from sql_agent.execution import execute_sql
from sql_agent.schema import inspect_schema, prompt_schema
from sql_agent.semantic_ir import GroundedIntent
from sql_agent.talon import gate, load_asset, repair
from sql_agent.v7_critic import select_candidate
from sql_agent.v7_data import open_cases
from sql_agent.v7_evaluation import V7Scoring

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
parser.add_argument("--resume", type=Path)
args = parser.parse_args()
root = args.artifacts.resolve()
eligibility = gate(root)
if not eligibility["eligible"]:
    raise SystemExit("Talon gate closed: " + "; ".join(eligibility["reasons"]))
asset, client = load_asset(root)
cases = open_cases(root, "calibration")
latest = root / "v7/sql-agent/eval/calibration/v7_three/latest.json"
run_id = json.loads(latest.read_text(encoding="utf-8"))["run_id"]
rows = [json.loads(line) for line in (latest.parent / run_id / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
by_id = {row["id"]: row for row in rows}
scorer = V7Scoring(cases)

def predict(question, context):
    original = by_id[next(case.id for case in cases if case.input == question and case.context == context)]
    selected, _ = select_candidate(original["candidates"])
    result = selected["result"]
    event = repair(client, question.split("Benchmark evidence:", 1)[0],
        prompt_schema(question, inspect_schema(Path(context["database"]))),
        json.dumps(original.get("evidence_packet", []), ensure_ascii=False),
        GroundedIntent.model_validate(original["grounding"]), selected["sql"], selected["critic"],
        selected["result_signals"])
    if event.get("accepted"):
        try:
            result = execute_sql(Path(context["database"]), event["sql"], timeout=10)
            return {**result, "output": result["rows"], "sql": event["sql"], "talon_event": event,
                    "api_charge_usd": 0, "input_tokens": event["input_tokens"], "output_tokens": event["output_tokens"]}
        except (ValueError, RuntimeError, TimeoutError) as error:
            event = {**event, "accepted": False, "execution_error": str(error)}
    return {**result, "output": result["rows"], "sql": selected["sql"], "talon_event": event,
            "api_charge_usd": 0, "input_tokens": 0, "output_tokens": 0}

metrics = scorer.metrics()[:2]
report = run_suite(cases, predict, metrics, root / "v7/sql-agent/eval/calibration/talon_critic",
    identity={**client.check(), "mode": "live", "execution_policy": {"id": "sql-readonly-v3", "timeout_seconds": 10},
              "candidate": "talon_critic", "prompt_version": "sql-v7-talon-critic-v1",
              "configuration": {"asset": asset, "trigger": "one actionable calibrated component"}},
    thresholds={"completion": {"min": .97}}, resume=args.resume,
    details=lambda used_cases, predictions: {
        "wrong_to_correct": sum(not scorer.correct(case, by_id[case.id]) and scorer.correct(case, prediction)
                                for case, prediction in zip(used_cases, predictions)),
        "correct_to_wrong": sum(scorer.correct(case, by_id[case.id]) and not scorer.correct(case, prediction)
                                for case, prediction in zip(used_cases, predictions)),
        "contamination_note": "BIRD ecosystem critic; exposed calibration only"})
print(json.dumps(report, indent=2))
