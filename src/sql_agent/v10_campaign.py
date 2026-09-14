"""Frozen V10 scoped generator captures for memory and fresh pilot roles."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from llm_evals import run_suite
from llm_evals.capture import TimedClient

from .agent import _full_schema_selection, error_category
from .execution import execute_sql
from .generation import MODEL_PROFILES
from .guardrails import validate
from .integrity import changed_databases, snapshot_database_hashes
from .m_schema import split_question_evidence
from .policy import POLICY
from .schema import inspect_schema, linked_schema
from .v7_evaluation import V7Scoring, structural_error_families
from .v7_generation import arctic_reference_direct
from .v8_evidence import EvidenceReconstructor
from .v9_reasoning import scope_packet
from .v10_data import open_memory_capture, open_pilot


PROMPT_VERSION = "sql-v10-frozen-v9-scoped-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--scope", choices=["memory", "pilot"], required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); root = args.artifacts.resolve()
    cases = open_memory_capture(root) if args.scope == "memory" else open_pilot(root)
    before = snapshot_database_hashes(cases)
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    client = TimedClient(model=MODEL_PROFILES["arctic_sql"]["model"])
    # V10 pilot databases already have checksum-matched cache-v2 assets from
    # V9; training databases use a separate V10 cache root.
    reconstructor = EvidenceReconstructor(root / ("v10" if args.scope == "memory" else "v8"))

    def predict(question, context):
        started = perf_counter(); database = Path(context["database"])
        natural, _ = split_question_evidence(question); schema = inspect_schema(database)
        rendered = linked_schema(natural, schema, False)
        selection = _full_schema_selection(natural, schema, rendered, 0)
        source = reconstructor.packet(database, natural, schema, mode="all")
        scoped = scope_packet(source)
        generated = arctic_reference_direct(client, natural, rendered, scoped.text)
        base = {"raw_model_response": generated.raw_text, "decoding": generated.decoding,
                "schema_selection": selection, "evidence_packet": [fact.model_dump() for fact in source.facts],
                "evidence_packet_identity": source.identity, "scoped_evidence_identity": scoped.identity,
                "input_tokens": generated.input_tokens, "output_tokens": generated.output_tokens,
                "generation_seconds": perf_counter() - started, "api_charge_usd": 0, "policy_id": POLICY["id"]}
        try:
            sql = validate(generated.sql); result = execute_sql(database, sql, timeout=10)
            return {**base, **result, "output": result["rows"], "status": "completed", "sql": sql,
                    "attempts": [{"attempt": 1, "path": "arctic_scoped", "sql": sql, "status": "executed"}],
                    "candidates": [{"path": "arctic_scoped", "sql": sql, "status": "executed", "result": result}],
                    "pipeline_profile": "v10_scoped_control", "needs_review": False}
        except (ValueError, RuntimeError, TimeoutError) as error:
            return {**base, "status": "failed", "eval_status": "failed", "sql": generated.sql,
                    "rows": [], "columns": [], "output": None, "candidates": [], "error": str(error),
                    "attempts": [{"attempt": 1, "path": "arctic_scoped", "sql": generated.sql,
                                  "status": "rejected", "category": error_category(error)}],
                    "pipeline_profile": "v10_scoped_control", "needs_review": True}

    def details(used_cases, predictions):
        return {"database_hashes_before": before, "database_hash_mismatches": changed_databases(before),
                "model_call_timings": client.timings,
                "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
                "scope": f"{len(cases)} V10 {args.scope} cases; single frozen scoped generator",
                "api_cost_usd": 0, "locked_final_opened": False}

    identity = client.check()
    report = run_suite(cases, predict, scorer.metrics(), root / f"v10/sql-agent/eval/{args.scope}/scoped_control",
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "candidate": "v10_a_scoped_control", "prompt_version": PROMPT_VERSION,
                  "scope_version": "sql-v9-evidence-scope-v1", "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details, resume=args.resume,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"]})
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
