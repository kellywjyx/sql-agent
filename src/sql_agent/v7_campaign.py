"""V7 exposed-data captures; the V6 locked final is intentionally unreachable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from llm_evals import run_suite
from llm_evals.capture import TimedClient

from .agent import SQLAgent, _full_schema_selection, error_category
from .execution import execute_sql
from .generation import MODEL_PROFILES
from .guardrails import validate
from .integrity import changed_databases, snapshot_database_hashes
from .m_schema import split_question_evidence
from .policy import POLICY
from .schema import inspect_schema, linked_schema
from .semantic_ir import extract_intent, ground_intent
from .v7_evaluation import V7Scoring, structural_error_families
from .v7_generation import PROMPT_VERSION, arctic_reference_direct
from .v7_data import open_cases
from .value_index import ValueIndex
from .value_profiles import ValueProfiler


MODES = {
    "v6_arctic_oracle": {"pipeline": "v6", "model": "arctic", "evidence": "oracle", "paths": ["arctic_direct"]},
    "reference_oracle": {"pipeline": "reference", "model": "arctic", "evidence": "oracle", "paths": ["arctic_direct"]},
    "reference_none": {"pipeline": "reference", "model": "arctic", "evidence": "none", "paths": ["arctic_direct"]},
    "reference_generated": {"pipeline": "reference", "model": "arctic", "evidence": "generated", "paths": ["arctic_direct"]},
    "reference_q5_oracle": {"pipeline": "reference", "model": "arctic_q5", "evidence": "oracle",
                              "paths": ["arctic_direct"], "diagnostic": "q5_matched_20"},
    "v7_three": {"pipeline": "v7", "model": "mixed", "evidence": "generated",
                  "paths": ["arctic_direct", "arctic_ir", "qwen_decomposed"]},
    "v7_public": {"pipeline": "v7", "model": "mixed", "evidence": "generated",
                   "paths": ["arctic_direct", "arctic_ir"]},
    "qwen_v5": {"pipeline": "v5", "model": "qwen", "evidence": "oracle", "paths": ["qwen_v5"]},
}


def _prepare_assets(root: Path, cases, *, profiles: bool = False):
    indexes, seen = [], set()
    value_index = ValueIndex(root)
    profiler = ValueProfiler(root)
    for case in cases:
        database = Path(case.context["database"]).resolve()
        if database in seen:
            continue
        seen.add(database)
        schema = inspect_schema(database)
        indexes.append(value_index.build(database, schema))
        if profiles:
            profiler.build(database, schema)
    return indexes


def _reference_predict(database: Path, question: str, root: Path, client, evidence_mode: str) -> dict:
    started = perf_counter()
    schema = inspect_schema(database)
    natural, oracle = split_question_evidence(question)
    rendered = linked_schema(natural, schema, False)
    selection = _full_schema_selection(natural, schema, rendered, 0)
    packet = ValueIndex(root).evidence_packet(database, natural, schema, selection,
        oracle_evidence=oracle, evidence_mode=evidence_mode)
    generated = arctic_reference_direct(client, natural, rendered, packet.text)
    generation_seconds = perf_counter() - started
    try:
        sql = validate(generated.sql)
        result = execute_sql(database, sql, timeout=10)
        candidate = {"path": "arctic_direct", "sql": sql, "status": "executed", "result": result}
        return {**result, "output": result["rows"], "status": "completed", "sql": sql,
            "attempts": [{"attempt": 1, "path": "arctic_direct", "sql": sql, "status": "executed",
                          "raw_model_response": generated.raw_text}], "candidates": [candidate],
            "raw_model_response": generated.raw_text, "decoding": generated.decoding,
            "schema_selection": selection, "evidence_packet": packet.facts,
            "evidence_packet_identity": packet.identity, "pipeline_profile": "reference_like",
            "input_tokens": generated.input_tokens, "output_tokens": generated.output_tokens,
            "generation_seconds": generation_seconds, "api_charge_usd": 0,
            "policy_id": POLICY["id"], "needs_review": False}
    except (ValueError, RuntimeError, TimeoutError) as error:
        return {"status": "failed", "eval_status": "failed", "sql": generated.sql, "rows": [], "columns": [],
            "output": None, "attempts": [{"attempt": 1, "path": "arctic_direct", "sql": generated.sql,
                "status": "rejected", "category": error_category(error), "error": str(error)[:1500],
                "raw_model_response": generated.raw_text}], "candidates": [],
            "error": str(error), "raw_model_response": generated.raw_text, "decoding": generated.decoding,
            "schema_selection": selection, "evidence_packet": packet.facts,
            "evidence_packet_identity": packet.identity, "pipeline_profile": "reference_like",
            "input_tokens": generated.input_tokens, "output_tokens": generated.output_tokens,
            "generation_seconds": generation_seconds, "api_charge_usd": 0, "policy_id": POLICY["id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--stage", choices=["reproduction", "calibration", "selection", "regression"], required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pilot-size", type=int)
    parser.add_argument("--prepare-assets", action="store_true")
    args = parser.parse_args()
    root, config = args.artifacts.resolve(), MODES[args.mode]
    allowed = {
        "reproduction": {"v6_arctic_oracle", "reference_oracle", "reference_none", "reference_generated",
                         "reference_q5_oracle"},
        "calibration": {"v7_three"}, "selection": {"v7_three", "qwen_v5"},
        "regression": {"v7_public", "qwen_v5"},
    }
    if args.mode not in allowed[args.stage]:
        parser.error(f"Mode {args.mode} is not valid for {args.stage}")
    if args.mode == "reference_q5_oracle" and args.pilot_size != 20:
        parser.error("The approved Q5 diagnostic is restricted to exactly --pilot-size 20")
    cases = open_cases(root, args.stage)
    if args.pilot_size:
        if not 1 <= args.pilot_size <= len(cases):
            parser.error("--pilot-size is outside the exposed role")
        cases = cases[:args.pilot_size]
    if args.prepare_assets:
        print(json.dumps(_prepare_assets(root, cases, profiles=config["pipeline"] == "v6"), indent=2)); return
    before = snapshot_database_hashes(cases)
    official_scorer_path = root / "v2/sql-agent/bird/downloads/evaluation_ex.py"
    scorer = V7Scoring(cases, official_scorer_path=official_scorer_path)
    model_name = (MODEL_PROFILES["arctic_sql_q5"]["model"] if config["model"] == "arctic_q5" else
                  MODEL_PROFILES["arctic_sql"]["model"] if config["model"] in {"arctic", "mixed"} else
                  MODEL_PROFILES["qwen_v5"]["model"])
    primary = TimedClient(model=model_name)
    qwen = primary if config["model"] == "qwen" else TimedClient(model=MODEL_PROFILES["qwen_v5"]["model"])
    clients = {"arctic_sql": primary, "qwen_v5": qwen}
    agent_cache = {}

    def predict(question, context):
        database = Path(context["database"])
        if config["pipeline"] == "reference":
            return _reference_predict(database, question, root, primary, config["evidence"])
        key = str(database.resolve())
        if key not in agent_cache:
            agent_cache[key] = SQLAgent(database, qwen, max_attempts=2, execution_timeout=10,
                                        artifacts=root, clients=clients)
        agent = agent_cache[key]
        if config["pipeline"] == "v6":
            return agent.ask(question, schema_mode="full", revised_linking=True, correction=True,
                model_profile="arctic_sql", generation_strategy="single", value_mode="profiled",
                prompt_style="direct", pipeline_profile="v5")
        if config["pipeline"] == "v5":
            return agent.ask(question, schema_mode="auto", revised_linking=True, correction=True,
                model_profile="qwen_v5", generation_strategy="single", value_mode="probe",
                prompt_style="direct", pipeline_profile="v5")
        return agent.ask(question, schema_mode="auto", correction=True, pipeline_profile="v7_grounded",
            generation_strategy="adaptive_two" if len(config["paths"]) == 2 else "single",
            evidence_mode=config["evidence"], candidate_paths=config["paths"], capture_candidate_results=True)

    def details(used_cases, predictions):
        evidence = scorer.evidence_metrics(used_cases, predictions)
        return {**evidence, "reference_errors": scorer.errors,
            "database_hashes_before": before, "database_hash_mismatches": changed_databases(before),
            "model_call_timings": primary.timings + ([] if qwen is primary else qwen.timings),
            "generation_seconds_total": sum(row.get("generation_seconds", 0) for row in predictions),
            "grounding_seconds_total": sum(row.get("grounding_seconds", 0) for row in predictions),
            "needs_review_count": sum(bool(row.get("needs_review")) for row in predictions),
            "official_scorer": {"commit": "b3d4bcbbae9a96934ad812551eb400c7a3b23c12",
                "sha256": "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89",
                "compatibility": "checksum-verified calculate_ex function executed in isolation"},
            "scope": "Exposed V7 development material; V6 locked final remains unopened"}

    identity = primary.check()
    report = run_suite(cases, predict, scorer.metrics(), root / f"v7/sql-agent/eval/{args.stage}/{args.mode}",
        identity={**identity, "mode": "live", "execution_policy": {**POLICY, "timeout_seconds": 10},
            "candidate": args.mode, "prompt_version": PROMPT_VERSION,
            "configuration": config, "evidence_mode": config["evidence"], "api_cost_budget_usd": 0},
        thresholds={"execution_accuracy_v7": {"min": 0},
                    "official_gold_self_consistency": {"min": .99}},
        details=details, resume=args.resume,
        analyze=lambda case, prediction: {"category": "reference_failure" if case.id in scorer.errors else
            "application_failure" if prediction.get("status") != "completed" else
            "correct" if scorer.correct(case, prediction) else "wrong_result",
            "error_families": structural_error_families(case.expected["sql"], prediction.get("sql")),
            "expected_sql": case.expected["sql"], "actual_sql": prediction.get("sql"),
            "db_id": case.metadata["db_id"]})
    report["details"]["raw_capture_sha256"] = hashlib.sha256(
        (Path(report["output_dir"]) / "predictions.jsonl").read_bytes()).hexdigest()
    Path(report["output_dir"], "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["run_id"], report["metrics"], flush=True)


if __name__ == "__main__":
    main()
