"""Matched V9 single-path experiments on already exposed development cases."""
from __future__ import annotations

import argparse
import json
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
from .v8_data import open_cases, open_pilot
from .v8_evidence import EvidenceReconstructor
from .v9_data import prepare
from .v9_probes import verify
from .v9_reasoning import SCOPE_VERSION, scope_packet
from .v9_utilization import METRIC_VERSION as UTILIZATION_VERSION, aggregate, analyze


PROMPT_VERSION = "sql-v9-execution-grounded-v1"
MODES = {"p0_cache_v2", "scoped", "probed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--scope", choices=["pilot", "development"], default="pilot")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    protocol = prepare(root)
    if args.scope == "development":
        decision_path = root / "v9/sql-agent/pilot-decision.json"
        if not decision_path.exists():
            parser.error("V9 pilot decision is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not decision.get("development_eligible") or args.mode != decision.get("selected"):
            parser.error("Only the frozen pilot winner may run on exposed development")
        cases = open_cases(root)
    else:
        cases = open_pilot(root)
    before = snapshot_database_hashes(cases)
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    client = TimedClient(model=MODEL_PROFILES["arctic_sql"]["model"])
    reconstructor = EvidenceReconstructor(root / "v8")

    def predict(question, context):
        started = perf_counter()
        database = Path(context["database"])
        natural, _ = split_question_evidence(question)
        schema = inspect_schema(database)
        rendered = linked_schema(natural, schema, False)
        selection = _full_schema_selection(natural, schema, rendered, 0)
        source = reconstructor.packet(database, natural, schema, mode="all")
        facts = [fact.model_dump() for fact in source.facts]
        scoped = None; verified = None
        if args.mode in {"scoped", "probed"}:
            scoped = scope_packet(source)
            evidence_text = scoped.text
        else:
            evidence_text = source.text
        if args.mode == "probed":
            verified = verify(database, scoped)
            evidence_text = verified.text
        generated = arctic_reference_direct(client, natural, rendered, evidence_text)
        base = {
            "raw_model_response": generated.raw_text, "decoding": generated.decoding,
            "schema_selection": selection, "evidence_packet": facts,
            "evidence_packet_identity": source.identity, "scoped_evidence_identity": scoped.identity if scoped else None,
            "probe_identity": verified.identity if verified else None,
            "probe_observations": [row.model_dump() for row in verified.observations] if verified else [],
            "evidence_mode": args.mode, "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens, "generation_seconds": perf_counter() - started,
            "api_charge_usd": 0, "policy_id": POLICY["id"],
        }
        try:
            sql = validate(generated.sql)
            result = execute_sql(database, sql, timeout=10)
            return {**base, **result, "output": result["rows"], "status": "completed", "sql": sql,
                    "attempts": [{"attempt": 1, "path": "arctic_direct", "sql": sql, "status": "executed"}],
                    "candidates": [{"path": "arctic_direct", "sql": sql, "status": "executed", "result": result}],
                    "pipeline_profile": f"v9_{args.mode}", "needs_review": False}
        except (ValueError, RuntimeError, TimeoutError) as error:
            return {**base, "status": "failed", "eval_status": "failed", "sql": generated.sql,
                    "rows": [], "columns": [], "output": None, "candidates": [], "error": str(error),
                    "attempts": [{"attempt": 1, "path": "arctic_direct", "sql": generated.sql,
                                  "status": "rejected", "category": error_category(error)}],
                    "pipeline_profile": f"v9_{args.mode}", "needs_review": True}

    def details(used_cases, predictions):
        utilization = [analyze(case.expected["sql"], prediction.get("sql"),
                               prediction.get("evidence_packet", []))
                       for case, prediction in zip(cases, predictions)]
        return {"evidence_utilization": aggregate(utilization),
                "utilization_by_case": {case.id: value for case, value in zip(cases, utilization)},
                "database_hashes_before": before, "database_hash_mismatches": changed_databases(before),
                "model_call_timings": client.timings,
                "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
                "scope": f"{len(cases)} exposed V9 {args.scope} cases; locked final unopened",
                "api_cost_usd": 0}

    identity = client.check()
    report = run_suite(cases, predict, scorer.metrics(), root / f"v9/sql-agent/eval/{args.scope}/{args.mode}",
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "candidate": args.mode, "prompt_version": PROMPT_VERSION,
                  "scope_version": SCOPE_VERSION if args.mode != "p0_cache_v2" else None,
                  "utilization_metric_version": UTILIZATION_VERSION,
                  "knowledge_cache": protocol["cache_control"], "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details, resume=args.resume,
        analyze=lambda case, prediction: {
            "category": "reference_failure" if case.id in scorer.errors else
                        "application_failure" if prediction.get("status") != "completed" else
                        "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"]})
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
