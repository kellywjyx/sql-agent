"""Guarded local BIRD comparisons. Gold SQL is accessed only by scoring."""
import argparse
import json
from pathlib import Path

from llm_evals import Metric, load_cases, run_suite
from llm_evals.dataset import fingerprint, write_json
from llm_evals.local import OllamaClient

from .agent import SQLAgent
from .execution import execute_sql


def evaluate(root, split):
    target = root / "v2/sql-agent"
    cases = load_cases(target / "bird" / f"{split}.jsonl")
    manifest = json.loads((target / "bird/manifest.json").read_text(encoding="utf-8"))
    client = OllamaClient()
    identity = client.check()
    gold_results, gold_errors = {}, {}
    # This stage never passes gold SQL into SQLAgent or the model prompt.
    for case in cases:
        try:
            gold_results[case.id] = execute_sql(Path(case.context["database"]), case.expected["sql"], timeout=10)
        except (ValueError, TimeoutError) as error:
            gold_errors[case.id] = str(error)
    reports = []
    for setting, linking, correction in [("baseline", False, False), ("linking", True, False), ("corrected", True, True)]:
        def ex(case, prediction):
            if prediction["status"] != "completed" or case.id not in gold_results:
                return 0.
            # Matches the pinned upstream calculate_ex set-of-rows comparator;
            # unlike local duplicate-preserving EX, duplicates/order are ignored.
            return float({tuple(row) for row in gold_results[case.id]["rows"]} ==
                         {tuple(row) for row in prediction.get("rows", [])})
        def accuracy(cs, ps):
            return sum(ex(c, p) for c, p in zip(cs, ps)) / len(cs)
        results = []
        def predict(question, context):
            try:
                result = SQLAgent(Path(context["database"]), client, execution_timeout=10).ask(question, linking=linking, correction=correction)
            except Exception:
                results.append({"status": "failed", "eval_status": "failed", "rows": []})
                raise
            results.append(result)
            return result
        metrics = [Metric("guarded_bird_ex", accuracy, version="2", bootstrap=True),
                   Metric("first_attempt_accuracy", lambda cs, ps: sum(ex(c, p) for c, p in zip(cs, ps)
                          if len(p.get("attempts", [])) == 1) / len(cs), version="2", bootstrap=True),
                   Metric("first_attempt_execution_rate", lambda cs, ps: sum(len(p.get("attempts", [])) == 1 and p["status"] == "completed" for p in ps) / len(cs), version="2"),
                   Metric("execution_recovered_fraction", lambda cs, ps: sum(len(p.get("attempts", [])) > 1 and p["status"] == "completed" for p in ps) / len(cs), version="2"),
                   Metric("timeout_case_fraction", lambda cs, ps: sum(any(a.get("category") == "timeout" for a in p.get("attempts", [])) for p in ps) / len(cs), version="2")]
        def details():
            by_difficulty = {}
            for difficulty in sorted({c.metadata["difficulty"] for c in cases}):
                subset = [(c, {**p, "status": "completed" if p.get("eval_status") != "failed" else "failed"})
                          for c, p in zip(cases, results) if c.metadata["difficulty"] == difficulty]
                by_difficulty[difficulty] = {"count": len(subset), "execution_accuracy": sum(ex(c, p) for c, p in subset) / len(subset)}
            initial_errors = sum(bool(p.get("attempts")) and p["attempts"][0]["status"] != "executed" for p in results)
            recovered = sum(len(p.get("attempts", [])) > 1 and p.get("status") == "completed" for p in results)
            return {"by_difficulty": by_difficulty, "gold_execution_errors": gold_errors,
                    "correction_recovery": {"initial_execution_failures": initial_errors, "recovered": recovered,
                                            "conditional_rate": recovered / initial_errors if initial_errors else None},
                    "claim": "Local guarded Mini-Dev evaluation using upstream EX comparator; not an official leaderboard submission"}
        report = run_suite(cases, predict, metrics, target / "eval" / split / setting,
                           identity={**identity, "mode": "live", "setting": setting, "split": split,
                                     "scorer": manifest["scorer"], "prompt_version": "sql-v2-schema-graph-values",
                                     "configuration_hash": fingerprint({"linking": linking, "correction": correction,
                                                                         "timeout_seconds": 10, "max_rows": 10000,
                                                                         "context_tokens": 8192, "max_new_tokens": 400})},
                           analyze=lambda c, p: {"status": p["status"], "category": "gold_execution_failure" if c.id in gold_errors else
                                                "application_failure" if p["status"] != "completed" else "execution_mismatch",
                                                "correct": bool(ex(c, p)), "expected_sql": c.expected["sql"],
                                                "actual_sql": p.get("sql"), "difficulty": c.metadata["difficulty"]}
                                   if not ex(c, p) else None, details=details)
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    write_json(target / f"{split}-campaign.json", {"reports": reports, "all_cases_attempted": True,
               "quality_gate_passed": all(r["status"] == "completed" for r in reports)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=["development", "final"], default="development")
    args = parser.parse_args()
    evaluate(args.artifacts.resolve(), args.split)
