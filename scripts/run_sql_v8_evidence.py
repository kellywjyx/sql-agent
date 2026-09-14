"""Build V8 caches and run the CPU-only evidence-reconstruction gate."""
import argparse
import hashlib
import json
from pathlib import Path

from sql_agent.schema import inspect_schema
from sql_agent.v8_data import open_cases
from sql_agent.v8_decomposition import build as build_decomposition
from sql_agent.v8_evaluation import evidence_metrics, load_decomposition
from sql_agent.v8_evidence import EvidenceReconstructor
from sql_agent.v8_knowledge import KnowledgeCache


parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path("../artifacts"))
parser.add_argument("--force-cache", action="store_true")
args = parser.parse_args()
root = args.artifacts.resolve()
build_decomposition(root)
cases = open_cases(root)
decomposition = load_decomposition(root / "v8/sql-agent/decomposition.jsonl")
cache = KnowledgeCache(root / "v8")
reconstructor = EvidenceReconstructor(root / "v8")
databases = {}
for case in cases:
    database = Path(case.context["database"])
    databases[str(database)] = database
manifests = []
for database in databases.values():
    manifests.append(cache.build(database, inspect_schema(database), force=args.force_cache))

target = root / "v8/sql-agent/evidence"
target.mkdir(parents=True, exist_ok=True)
all_reports = {}
for mode in ["literals", "descriptions", "normalization", "all"]:
    rows = []
    for case in cases:
        database = Path(case.context["database"])
        packet = reconstructor.packet(database, case.input, inspect_schema(database), mode=mode)
        rows.append({"id": case.id, "mode": mode, "identity": packet.identity,
                     "facts": [fact.model_dump() for fact in packet.facts], "text": packet.text,
                     "byte_count": packet.byte_count, "candidates_considered": packet.candidates_considered,
                     "probe_count": packet.probe_count})
    output = target / f"{mode}.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    overall = evidence_metrics(cases, rows, decomposition)
    group_a_pairs = [(case, row, label) for case, row, label in zip(cases, rows, decomposition)
                     if label["group"] == "A"]
    group_a = evidence_metrics([item[0] for item in group_a_pairs], [item[1] for item in group_a_pairs],
                               [item[2] for item in group_a_pairs])
    all_reports[mode] = {"overall": overall, "oracle_help_group_a": group_a,
                         "packet_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}

protocol = json.loads((root / "v8/sql-agent/protocol.json").read_text(encoding="utf-8"))
metrics = all_reports["all"]["oracle_help_group_a"]
gate = {"literal_recall": metrics["literal_recall"] >= protocol["gates"]["literal_recall"],
        "value_column_accuracy": metrics["value_column_accuracy"] >= protocol["gates"]["value_column_accuracy"],
        "evidence_precision": metrics["evidence_precision"] >= protocol["gates"]["evidence_precision"]}
report = {"version": "sql-v8-cpu-evidence-gate-v1", "cases": len(cases), "databases": len(databases),
          "cache_identities": [row["identity"] for row in manifests], "modes": all_reports,
          "gate_b": gate, "gate_b_passed": all(gate.values()),
          "model_inference_permitted": all(gate.values()), "locked_final_opened": False,
          "api_cost_usd": 0}
(target / "gate-b.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"gate_b": gate, "gate_b_passed": all(gate.values()),
                  "group_a_metrics": metrics}, indent=2))
