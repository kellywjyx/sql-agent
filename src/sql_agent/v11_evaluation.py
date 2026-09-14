"""Base-versus-adapter V11 evaluation under an identical local runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_evals import Metric, load_cases, run_suite
from llm_evals.campaign import open_dataset

from .integrity import changed_databases, snapshot_database_hashes
from .policy import POLICY
from .semantic_ir import reference_structure
from .v7_evaluation import METRIC_VERSION as BASE_METRIC_VERSION, V7Scoring, structural_error_families
from .v11_data import open_role
from .v11_runtime import V11Runtime, ask
from .v8_selection import _interval, _run


METRIC_VERSION = "sql-v11-metrics-v1"


def _cases(root: Path, stage: str):
    if stage == "pilot":
        return open_role(root, "development")[:60]
    if stage == "development":
        return open_role(root, "development")
    prepared = root / "v11/sql-agent/data/eval-regression.jsonl"
    if not prepared.is_file():
        raise RuntimeError("V11 regression inputs are not prepared. Run: sql-agent v11-prepare-eval --stage regression")
    return open_dataset(prepared, purpose="inspected_regression",
                        ledger=root / "v11/exposure.jsonl")


def _adapter(root: Path, stage: str, variant: str) -> Path | None:
    if variant == "base": return None
    if stage == "pilot": return root / f"v11/sql-agent/training/pilot/{variant}/final"
    decision = json.loads((root / "v11/sql-agent/pilot-decision.json").read_text(encoding="utf-8"))
    return root / f"v11/sql-agent/training/full/{decision['selected_recipe']}/final"


def component_metrics(cases, predictions) -> dict:
    names = ("projection", "predicate", "join", "aggregate", "grain", "ordering")
    values = {name: [] for name in names}
    for case, prediction in zip(cases, predictions):
        failures = set(structural_error_families(case.expected["sql"], prediction.get("sql")))
        for name in names: values[name].append(name not in failures)
    return {f"structural_{name}_accuracy": sum(rows) / len(rows) if rows else 0
            for name, rows in values.items()}


def run(root: Path, stage: str, variant: str, *, resume: Path | None = None) -> dict:
    cases = _cases(root, stage); before = snapshot_database_hashes(cases)
    adapter = _adapter(root, stage, variant); runtime = V11Runtime(root, adapter)
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    metrics = [*scorer.metrics(), *[Metric(name, lambda cs, ps, key=name: component_metrics(cs, ps)[key],
              version=METRIC_VERSION) for name in [f"structural_{part}_accuracy" for part in
              ("projection", "predicate", "join", "aggregate", "grain", "ordering")]]]
    def details(_, predictions):
        value = {"database_hashes_before": before,
            "database_hash_mismatches": changed_databases(before), "model_call_timings": runtime.timings,
            "cold_seconds": runtime.timings[0] if runtime.timings else None,
            "warm_seconds": runtime.timings[1:] if len(runtime.timings) > 1 else [],
            "peak_gpu_bytes": __import__("torch").cuda.max_memory_allocated(),
            "api_cost_usd": 0, "locked_final_opened": False}
        if stage == "regression":
            _, historical = _run(root, root / "v5/sql-agent/eval/final/coder_full")
            current = {case.id: scorer.correct(case, prediction) for case, prediction in zip(cases, predictions)}
            value["comparison_vs_v5"] = _interval(cases,
                {case.id: historical[case.id] for case in cases}, current, seed=20260923)
        return value

    report = run_suite(cases, lambda question, context: ask(runtime, root, Path(context["database"]), question,
            prepared_messages=context.get("messages")),
        metrics, root / f"v11/sql-agent/eval/{stage}/{variant}",
        identity={**runtime.identity(), "mode": "live", "variant": variant,
                  "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "metric_version": METRIC_VERSION, "base_metric_version": BASE_METRIC_VERSION,
                  "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"], "query_shape": case.metadata.get("query_shape")}, resume=resume)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "development", "regression"], required=True)
    parser.add_argument("--variant", choices=["base", "A", "B", "selected"], required=True)
    parser.add_argument("--resume", type=Path); args = parser.parse_args()
    report = run(args.artifacts.resolve(), args.stage, args.variant, resume=args.resume)
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__": main()
