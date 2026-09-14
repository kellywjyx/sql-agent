"""Inspect frozen BIRD outcomes without executing SQL or calling the model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from llm_evals.runner import latest_run


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_category(reason):
    text = reason.lower()
    if "too many columns" in text:
        return "schema_column_limit"
    if "not authorized to use function" in text:
        return "function_authorizer"
    if "interrupted" in text or "deadline" in text:
        return "deadline"
    if "exceeds" in text and ("rows" in text or "budget" in text):
        return "result_budget"
    if "no such table" in text or "no such column" in text:
        return "schema_reference_mismatch"
    return "other_reference_failure"


def classify(predictions, failures, gold_errors):
    ids = [p["id"] for p in predictions]
    wrong = [f["id"] for f in failures]
    if len(set(ids)) != len(ids) or len(set(wrong)) != len(wrong):
        raise ValueError("Duplicate prediction or failure IDs")
    if not set(wrong).issubset(ids) or not set(gold_errors).issubset(wrong):
        raise ValueError("Unaligned failure or gold-error IDs")
    if any(f.get("correct") is not False for f in failures):
        raise ValueError("Failure analysis lacks explicit correctness verdicts")
    wrong = set(wrong)
    rows = []
    for prediction in predictions:
        key = prediction["id"]
        executed = prediction["status"] == "completed"
        if not executed and key not in wrong:
            raise ValueError("Failed execution is absent from failure analysis")
        attempts = prediction.get("attempts", [])
        rejected = [a.get("category", "uncategorized") for a in attempts if a["status"] != "executed"]
        category = ("gold_unscorable" if key in gold_errors else "application_failure" if not executed
                    else "execution_mismatch" if key in wrong else "correct")
        rows.append({"id": key, "category": category, "executed": executed,
                     "attempt_count": len(attempts), "rejected_attempt_categories": rejected,
                     "uncaught_error_type": prediction.get("error", "unknown").split(":", 1)[0] if not attempts and not executed else None,
                     "execution_recovered": executed and len(attempts) > 1,
                     "correct": key not in wrong, "empty_execution": executed and not prediction.get("rows", [])})
    return rows


def analyze(root, split):
    target = root / "v2/sql-agent"
    expected_ids = [case["id"] for case in jsonl(target / "bird" / f"{split}.jsonl")]
    runs, details = {}, []
    for mode in ["baseline", "linking", "corrected"]:
        folder = latest_run(target / "eval" / split / mode)
        report = read(folder / "metrics.json")
        predictions = jsonl(folder / "predictions.jsonl")
        if expected_ids != [p["id"] for p in predictions] or report["count"] != len(predictions):
            raise ValueError("Incomplete or unaligned BIRD predictions")
        rows = classify(predictions, jsonl(folder / "failure-analysis.jsonl"), report["details"]["gold_execution_errors"])
        accuracy = sum(row["correct"] for row in rows) / len(rows)
        if abs(accuracy - report["metrics"]["guarded_bird_ex"]) > 1e-9:
            raise ValueError("Failure verdicts disagree with recorded accuracy")
        recovered = [row for row in rows if row["execution_recovered"]]
        runs[mode] = {"run_id": report["run_id"], "dataset_hash": report["dataset_hash"],
                      "predictions_sha256": hashlib.sha256((folder / "predictions.jsonl").read_bytes()).hexdigest(),
                      "count": len(rows), "outcomes": dict(Counter(row["category"] for row in rows)),
                      "rejected_attempt_categories": dict(Counter(c for row in rows for c in row["rejected_attempt_categories"])),
                      "uncaught_error_types": dict(Counter(row["uncaught_error_type"] for row in rows if row["uncaught_error_type"])),
                      "initial_execution_failures": sum(bool(row["rejected_attempt_categories"]) for row in rows),
                      "recovered_execution": len(recovered), "recovered_correct": sum(row["correct"] for row in recovered),
                      "recovered_gold_unscorable": sum(row["category"] == "gold_unscorable" for row in recovered),
                      "empty_executions": sum(row["empty_execution"] for row in rows),
                      "correct_empty_executions": sum(row["empty_execution"] and row["correct"] for row in rows),
                      "gold_failure_categories": dict(Counter(gold_category(reason) for reason in report["details"]["gold_execution_errors"].values())),
                      "gold_execution_errors": report["details"]["gold_execution_errors"]}
        details.extend({"setting": mode, **row} for row in rows)
    if len({run["dataset_hash"] for run in runs.values()}) != 1:
        raise ValueError("BIRD analysis datasets differ")
    output = target / "failure-review" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    summary = {"version": "sql-diagnostics-1", "split": split, "runs": runs,
               "note": "Execution recovery is not answer correctness. Gold execution failures retain zero credit in the full denominator; they are not solely model errors."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details), encoding="utf-8")
    lines = [f"# BIRD {split} failure analysis", "", summary["note"], "",
             "| Setting | Correct | Wrong executed answer | Application failure | Gold unscorable | Recovered execution / correct |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode, run in runs.items():
        outcomes = run["outcomes"]
        lines.append("| " + " | ".join([mode, *[str(outcomes.get(k, 0)) for k in
                     ["correct", "execution_mismatch", "application_failure", "gold_unscorable"]],
                     f"{run['recovered_execution']} / {run['recovered_correct']}"]) + " |")
    lines += ["", "## Reference execution failures", ""]
    for category, count in runs["baseline"]["gold_failure_categories"].items():
        examples = [key for key, reason in runs["baseline"]["gold_execution_errors"].items() if gold_category(reason) == category][:5]
        lines.append(f"- {category}: {count} cases; example IDs: " + ", ".join(f"`{key}`" for key in examples) + ".")
    lines += ["", "Full per-case reference errors are retained in the companion JSON. The worker's existing SQLITE_LIMIT_COLUMN=64 can reject an otherwise readable wide schema with a misleading 'malformed schema / too many columns' message. This is a configured compatibility limit, not evidence that the dataset file is corrupt. CURRENT_TIMESTAMP is outside the function allowlist. These controls were not relaxed during the campaign."]
    lines += ["", "Outcome categories are mutually exclusive, with gold-unscorable taking precedence. A case can also have an application error; the separate attempt-category counts retain that information. Failed attempts count individually, while outcome rows count cases.",
              "", "Uncaught tokenizer exceptions can end a case before an attempt is recorded and before correction. They fail closed, but this is a recovery limitation of the frozen implementation. The companion JSON separates these exception types from ordinary rejected attempts.",
              "", "Empty executed results remain successful executions. Their correctness still depends on reference execution. Queries exceeding the row/byte budget fail; evaluation results are never silently truncated. Database mismatches and deadlines were not bypassed to obtain a score.",
              "", "This analysis reads the frozen verdicts; it does not rerun or repair gold SQL, model SQL, database schemas, or outputs. Final inspected examples become regression material for a future campaign.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (target / f"{split}-failure-review-latest.json").write_text(json.dumps({"directory": output.name}), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=["development", "final"], required=True)
    args = parser.parse_args()
    analyze(args.artifacts.resolve(), args.split)
