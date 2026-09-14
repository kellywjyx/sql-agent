"""Single-path Arctic Q4 evidence ablations on the exposed V8 pilot."""
from __future__ import annotations

import argparse
import hashlib
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
from .v8_evaluation import evidence_metrics, load_decomposition
from .v8_evidence import EVIDENCE_VERSION, EvidenceReconstructor


PROMPT_VERSION = "sql-v8-arctic-reference-like-v1"
MODES = {"generated_literals": "literals", "generated_descriptions": "descriptions",
         "generated_normalization": "normalization", "generated_all": "all"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--scope", choices=["pilot", "development"], default="pilot")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    gate = json.loads((root / "v8/sql-agent/evidence/gate-b.json").read_text(encoding="utf-8"))
    if not gate.get("gate_b_passed"):
        parser.error("V8 Gate B has not passed; model inference is blocked")
    if args.scope == "development":
        decision = json.loads((root / "v8/sql-agent/pilot-decision.json").read_text(encoding="utf-8"))
        if not decision.get("gate_c_passed") or args.mode != decision.get("selected"):
            parser.error("Only the frozen Gate-C winner may run on broader exposed development")
        cases = open_cases(root)
    else:
        cases = open_pilot(root)
    decomposition_all = load_decomposition(root / "v8/sql-agent/decomposition.jsonl")
    labels = {row["id"]: row for row in decomposition_all}
    decomposition = [labels[case.id] for case in cases]
    before = snapshot_database_hashes(cases)
    scorer = V7Scoring(cases, official_scorer_path=root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    client = TimedClient(model=MODEL_PROFILES["arctic_sql"]["model"])
    reconstructor = EvidenceReconstructor(root / "v8")
    packet_rows = []

    def predict(question, context):
        started = perf_counter()
        database = Path(context["database"])
        natural, _ = split_question_evidence(question)
        schema = inspect_schema(database)
        rendered = linked_schema(natural, schema, False)
        selection = _full_schema_selection(natural, schema, rendered, 0)
        packet = reconstructor.packet(database, natural, schema, mode=MODES[args.mode])
        packet_rows.append({"id": context["case_id"], "facts": [fact.model_dump() for fact in packet.facts]})
        generated = arctic_reference_direct(client, natural, rendered, packet.text)
        elapsed = perf_counter() - started
        base = {"raw_model_response": generated.raw_text, "decoding": generated.decoding,
                "schema_selection": selection, "evidence_packet": [fact.model_dump() for fact in packet.facts],
                "evidence_packet_identity": packet.identity, "evidence_mode": args.mode,
                "input_tokens": generated.input_tokens, "output_tokens": generated.output_tokens,
                "generation_seconds": elapsed, "api_charge_usd": 0, "policy_id": POLICY["id"]}
        try:
            sql = validate(generated.sql)
            result = execute_sql(database, sql, timeout=10)
            return {**base, **result, "output": result["rows"], "status": "completed", "sql": sql,
                    "attempts": [{"attempt": 1, "path": "arctic_direct", "sql": sql,
                                  "status": "executed", "raw_model_response": generated.raw_text}],
                    "candidates": [{"path": "arctic_direct", "sql": sql, "status": "executed", "result": result}],
                    "pipeline_profile": "v8_database_grounded", "needs_review": False}
        except (ValueError, RuntimeError, TimeoutError) as error:
            return {**base, "status": "failed", "eval_status": "failed", "sql": generated.sql,
                    "rows": [], "columns": [], "output": None, "candidates": [], "error": str(error),
                    "attempts": [{"attempt": 1, "path": "arctic_direct", "sql": generated.sql,
                                  "status": "rejected", "category": error_category(error),
                                  "error": str(error)[:1500], "raw_model_response": generated.raw_text}],
                    "pipeline_profile": "v8_database_grounded", "needs_review": True}

    # run_suite does not pass the case ID in context; inject a scoring-neutral
    # identifier into a private context copy solely to align packet metrics.
    run_cases = [case.model_copy(update={"context": {**case.context, "case_id": case.id}}) for case in cases]

    def details(used_cases, predictions):
        packets = [{"id": case.id, "facts": prediction.get("evidence_packet", [])}
                   for case, prediction in zip(cases, predictions)]
        return {"evidence_metrics": evidence_metrics(cases, packets, decomposition),
                "database_hashes_before": before, "database_hash_mismatches": changed_databases(before),
                "model_call_timings": client.timings,
                "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
                "scope": f"{len(cases)} exposed V8 {args.scope} cases; no oracle evidence; locked final unopened",
                "api_cost_usd": 0}

    identity = client.check()
    report = run_suite(run_cases, predict, scorer.metrics(), root / f"v8/sql-agent/eval/{args.scope}/{args.mode}",
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
                  "candidate": args.mode, "prompt_version": PROMPT_VERSION,
                  "evidence_version": EVIDENCE_VERSION, "evidence_mode": MODES[args.mode],
                  "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0}, "official_gold_self_consistency": {"min": .99}},
        details=details, resume=args.resume,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "db_id": case.metadata["db_id"], "oracle_group": labels[case.id]["group"]})
    output = Path(report["output_dir"])
    report["details"]["raw_capture_sha256"] = hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
