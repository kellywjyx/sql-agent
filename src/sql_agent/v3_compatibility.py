"""CPU-only saved-SQL replay; never relabel cross-policy changes as model gains."""
import argparse
import hashlib
import json
from pathlib import Path
from llm_evals import load_cases
from llm_evals.campaign import immutable_json
from llm_evals.runner import latest_run
from .execution import execute_sql
from .policy import POLICY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    old = root / "v2/sql-agent"
    import uuid
    output = root / "v3/sql-agent/compatibility" / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    cases = load_cases(old / "bird/final.jsonl")
    reference, errors = {}, {}
    with (output / "reference.jsonl").open("x", encoding="utf-8") as handle:
        for case in cases:
            try:
                reference[case.id] = execute_sql(Path(case.context["database"]), case.expected["sql"], timeout=10)
                row = {"id": case.id, "status": "completed", "row_count": reference[case.id]["row_count"]}
            except (ValueError, TimeoutError) as error:
                errors[case.id] = str(error)
                row = {"id": case.id, "status": "failed", "error": str(error)}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
    reports = []
    for mode in ["baseline", "linking", "corrected"]:
        path = latest_run(old / "eval/final" / mode) / "predictions.jsonl"
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [p["id"] for p in predictions] == [c.id for c in cases]
        counts = {"count": len(cases), "executed": 0, "no_saved_sql": 0, "correct_upstream_set_comparator": 0}
        with (output / f"{mode}.jsonl").open("x", encoding="utf-8") as handle:
            for case, prediction in zip(cases, predictions):
                sql = prediction.get("sql") or next((a["sql"] for a in reversed(prediction.get("attempts", [])) if a.get("sql")), None)
                row = {"id": case.id, "sql": sql}
                if not sql:
                    counts["no_saved_sql"] += 1
                    row["status"] = "no_saved_sql"
                else:
                    try:
                        result = execute_sql(Path(case.context["database"]), sql, timeout=10)
                        counts["executed"] += 1
                        correct = case.id in reference and {tuple(r) for r in reference[case.id]["rows"]} == {tuple(r) for r in result["rows"]}
                        counts["correct_upstream_set_comparator"] += correct
                        row.update(status="completed", correct=correct, row_count=result["row_count"])
                    except (ValueError, TimeoutError) as error:
                        row.update(status="failed", error=str(error))
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        reports.append({"mode": mode, **counts, "saved_predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(mode, counts, flush=True)
    immutable_json(output / "report.json", {"new_policy": {**POLICY, "timeout_seconds": 10}, "prior_policy": "SQLite schema/result column cap64",
        "scope": "v2 Mini-Dev inspected regression; compatibility analysis, no new model generation",
        "reference_coverage": len(reference) / len(cases), "reference_errors": errors, "modes": reports,
        "comparator": "v2 upstream set comparator retained to isolate policy effects; v3 fresh primary is duplicate/order preserving"})


if __name__ == "__main__":
    main()
