import argparse
import json
from pathlib import Path

from llm_evals import Metric, load_cases, run_suite

from .agent import SQLAgent
from .demo import create_demo
from .evaluation import execution_metric, load_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["seed", "ask", "evaluate", "import-benchmark", "index-schema", "inspect-schema",
                                                   "profile-values", "inspect-value-profile", "build-value-index",
                                                   "inspect-value-index", "inspect-evidence", "inspect-intent",
                                                   "build-v8-knowledge", "inspect-v8-evidence",
                                                   "inspect-v9-evidence", "verify-v9-evidence",
                                                   "v11-verify-asset", "v11-prepare-data", "v11-prepare-eval-assets", "v11-preflight",
                                                   "v11-reload", "v11-train-pilot", "v11-train-full",
                                                   "v11-evaluate", "v11-compare"])
    parser.add_argument("--database", type=Path, default=Path("artifacts/sql-agent/demo.sqlite"))
    parser.add_argument("--question", default="What is the total revenue from completed orders?")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "resources/demo.jsonl")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sql-agent/eval"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--schema-mode", choices=["auto", "full", "hybrid"], default="auto")
    parser.add_argument("--model-profile", choices=["default", "qwen_v5", "qwen_m_schema", "arctic_sql", "xiyan_sql"], default="default")
    parser.add_argument("--generation-strategy", choices=["single", "adaptive_two"], default="single")
    parser.add_argument("--value-mode", choices=["probe", "profiled"], default="probe")
    parser.add_argument("--prompt-style", choices=["direct", "plan_first"], default="direct")
    parser.add_argument("--pipeline-profile", choices=["v5", "v7_grounded", "v11_adapter"], default="v5")
    parser.add_argument("--recipe", choices=["A", "B"], default="A")
    parser.add_argument("--stage", choices=["pilot", "development", "regression"], default="pilot")
    parser.add_argument("--variant", choices=["base", "A", "B", "selected"], default="base")
    parser.add_argument("--attempt-id", default="attempt-1")
    parser.add_argument("--evidence-mode", choices=["none", "oracle", "generated"], default="generated")
    parser.add_argument("--v8-evidence-mode", choices=["literals", "descriptions", "normalization", "all"], default="all")
    parser.add_argument("--benchmark", choices=["bird", "spider"], default="bird")
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args()
    if args.command.startswith("v11-"):
        root = args.artifacts.resolve()
        if args.command == "v11-verify-asset":
            from .v11_assets import verify
            print(json.dumps(verify(root), indent=2)); return
        if args.command == "v11-prepare-data":
            from .v11_assets import model_path, verify
            from .v11_data import finalize
            verify(root)
            from transformers import AutoTokenizer
            print(json.dumps(finalize(root, AutoTokenizer.from_pretrained(model_path(root), local_files_only=True)), indent=2)); return
        if args.command == "v11-prepare-eval-assets":
            from .v11_data import prepare_evaluation_assets
            print(json.dumps(prepare_evaluation_assets(root, args.stage), indent=2)); return
        if args.command in {"v11-preflight", "v11-reload", "v11-train-pilot", "v11-train-full"}:
            from .v11_training import preflight, reload_check, train
            value = preflight(root, attempt_id=args.attempt_id) if args.command == "v11-preflight" else reload_check(root) if args.command == "v11-reload" else train(root, args.recipe, pilot=args.command == "v11-train-pilot")
            print(json.dumps(value, indent=2)); return
        if args.command == "v11-evaluate":
            from .v11_evaluation import run
            print(json.dumps(run(root, args.stage, args.variant), indent=2)); return
        from .v11_selection import freeze_development, freeze_pilot, freeze_regression
        function = {"pilot": freeze_pilot, "development": freeze_development, "regression": freeze_regression}[args.stage]
        print(json.dumps(function(root), indent=2)); return
    if args.command == "seed":
        create_demo(args.database)
        print(args.database.resolve())
        return
    if args.command == "import-benchmark":
        if args.database_root is None:
            parser.error("--database-root is required")
        cases = load_benchmark(args.dataset, args.database_root, args.benchmark)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8")
        return
    if args.command in {"index-schema", "inspect-schema"}:
        from .schema import inspect_schema
        from .schema_retrieval import HybridSchemaRetriever
        schema = inspect_schema(args.database)
        retriever = HybridSchemaRetriever(args.artifacts)
        value = retriever.build(args.database, schema) if args.command == "index-schema" else retriever.check(args.database, schema)
        print(json.dumps(value, indent=2))
        return
    if args.command in {"profile-values", "inspect-value-profile"}:
        from .schema import inspect_schema
        from .value_profiles import ValueProfiler
        schema = inspect_schema(args.database)
        profiler = ValueProfiler(args.artifacts)
        value = profiler.build(args.database, schema) if args.command == "profile-values" else profiler.check(args.database, schema)
        print(json.dumps(value, indent=2))
        return
    if args.command in {"build-v8-knowledge", "inspect-v8-evidence", "inspect-v9-evidence", "verify-v9-evidence"}:
        from .schema import inspect_schema
        from .v8_evidence import EvidenceReconstructor
        from .v8_knowledge import KnowledgeCache
        schema = inspect_schema(args.database)
        if args.command == "build-v8-knowledge":
            print(json.dumps(KnowledgeCache(args.artifacts).build(args.database, schema), indent=2)); return
        packet = EvidenceReconstructor(args.artifacts).packet(
            args.database, args.question, schema, mode=args.v8_evidence_mode)
        if args.command in {"inspect-v9-evidence", "verify-v9-evidence"}:
            from .v9_reasoning import scope_packet
            scoped = scope_packet(packet)
            if args.command == "verify-v9-evidence":
                from .v9_probes import verify
                verified = verify(args.database, scoped)
                print(json.dumps({"identity": verified.identity,
                                  "observations": [row.model_dump() for row in verified.observations],
                                  "text": verified.text,
                                  "database_hash_unchanged": verified.database_hash_unchanged}, indent=2)); return
            print(json.dumps({"identity": scoped.identity,
                              "facts": [row.model_dump() for row in scoped.facts],
                              "text": scoped.text, "bytes": scoped.byte_count}, indent=2)); return
        print(json.dumps({"identity": packet.identity, "mode": packet.mode,
                          "facts": [fact.model_dump() for fact in packet.facts],
                          "text": packet.text, "bytes": packet.byte_count,
                          "candidates_considered": packet.candidates_considered,
                          "probes": packet.probe_count}, indent=2)); return
    if args.command in {"build-value-index", "inspect-value-index", "inspect-evidence", "inspect-intent"}:
        from .schema import inspect_schema, prompt_schema
        from .value_index import ValueIndex
        from .semantic_ir import extract_intent, ground_intent
        schema = inspect_schema(args.database)
        index = ValueIndex(args.artifacts)
        if args.command == "build-value-index":
            print(json.dumps(index.build(args.database, schema), indent=2)); return
        if args.command == "inspect-value-index":
            print(json.dumps(index.check(args.database, schema), indent=2)); return
        rendered = prompt_schema(args.question, schema)
        selection = {"mode": "full", "selected_tables": [row["table"] for row in schema],
            "selected_columns": [{"table": row["table"], "column": column["name"]}
                                 for row in schema for column in row.get("column_details", [])],
            "join_paths": [[row["table"], key["table"]] for row in schema for key in row.get("foreign_keys", [])],
            "prompt_bytes": len(rendered.encode()), "retrieval_seconds": 0}
        packet = index.evidence_packet(args.database, args.question, schema, selection,
                                       evidence_mode=args.evidence_mode)
        if args.command == "inspect-evidence":
            print(json.dumps({"identity": packet.identity, "facts": packet.facts,
                              "bytes": packet.byte_count, "probes": packet.probe_count}, indent=2)); return
        agent = SQLAgent(args.database, artifacts=args.artifacts)
        intent, meta = extract_intent(agent._client_for("qwen_v5"), args.question, packet.facts)
        grounded = ground_intent(intent, schema, packet.facts, selection)
        print(json.dumps({"intent": intent.model_dump(), "grounding": grounded.model_dump(), "meta": meta}, indent=2)); return
    from .schema_retrieval import HybridSchemaRetriever
    retriever = HybridSchemaRetriever(args.artifacts) if args.schema_mode == "hybrid" else None
    agent = SQLAgent(args.database, artifacts=args.artifacts, schema_retriever=retriever)
    identity = agent.client.check()
    if args.command == "ask":
        print(json.dumps(agent.ask(args.question, schema_mode=args.schema_mode,
                                   revised_linking=True, semantic_review=False,
                                   model_profile=args.model_profile,
                                   generation_strategy=args.generation_strategy,
                                   value_mode=args.value_mode, prompt_style=args.prompt_style,
                                   pipeline_profile=args.pipeline_profile,
                                   evidence_mode=args.evidence_mode), indent=2))
        return
    cases = load_cases(args.dataset)
    for case in cases:
        if not case.context:
            case.context = {"database": str(args.database.resolve())}
    settings = [("baseline", False, False), ("linking", True, False), ("corrected", True, True)] if args.ablate else [("corrected", True, True)]
    reports = []
    for name, linking, correction in settings:
        report = run_suite(cases, lambda question, context: SQLAgent(Path(context["database"]), agent.client,
            artifacts=args.artifacts, schema_retriever=retriever).ask(
            question, linking=linking, correction=correction, schema_mode=args.schema_mode,
            revised_linking=True, semantic_review=True, model_profile=args.model_profile,
            generation_strategy=args.generation_strategy, value_mode=args.value_mode,
            prompt_style=args.prompt_style, pipeline_profile=args.pipeline_profile,
            evidence_mode=args.evidence_mode), [Metric("execution_accuracy", execution_metric)],
            args.output / name, identity={**identity, "mode": "live", "setting": name, "prompt_version": "sql-v1",
                                          "benchmark": "local duplicate-preserving EX; not official leaderboard scoring"})
        reports.append(report)
        print(json.dumps(report, indent=2))
    raise SystemExit(any(r["status"] != "completed" for r in reports))


if __name__ == "__main__":
    main()
