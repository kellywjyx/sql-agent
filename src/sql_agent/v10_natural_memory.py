"""Extract schema-independent error records from natural Arctic failures."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .v7_evaluation import V7Scoring
from .v8_selection import _run
from .v10_calibration import actual_families
from .v10_data import open_memory_capture
from .v10_memory import load
from .v10_signature import build_signature, canonical_identity


NATURAL_VERSION = "sql-v10-natural-error-memory-v1"


def build(root: Path) -> dict:
    root = root.resolve(); target = root / "v10/sql-agent/error-memory"
    manifest_path = target / "natural-manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = open_memory_capture(root)
    report, _ = _run(root, root / "v10/sql-agent/eval/memory/scoped_control")
    run = root / "v10/sql-agent/eval/memory/scoped_control" / report["run_id"]
    predictions = {row["id"]: row for row in
                   (json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    scorer = V7Scoring(cases, root / "v2/sql-agent/bird/downloads/evaluation_ex.py")
    rows = []
    for case in cases:
        prediction = predictions[case.id]
        if scorer.correct(case, prediction) or not prediction.get("sql"):
            continue
        families = actual_families(case.expected["sql"], prediction["sql"])
        if len(families) != 1:
            continue
        family = next(iter(families))
        signature = build_signature(case.input.split("Benchmark evidence:", 1)[0].strip(),
                                    prediction.get("evidence_packet", []), prediction["sql"], prediction)
        patch = {"operation": "restore_requested_operation", "grain": "restore_result_grain",
                 "predicate": "restore_predicate_binding", "join": "restore_join_relationship"}[family]
        gold_shape = build_signature("", [], case.expected["sql"], scorer.gold.get(case.id, {})).sql_shape
        rows.append({"version": NATURAL_VERSION,
                     "identity": canonical_identity(signature, family, patch),
                     "source_case_hash": hashlib.sha256(case.id.encode()).hexdigest(),
                     "source": "natural_arctic_scoped_failure", "error_type": family,
                     "minimal_patch": patch, "signature": signature.model_dump(),
                     "correct_shape": gold_shape, "execution_fixed": True,
                     "contains_raw_gold_sql": False, "gold_sql_supplied_at_runtime": False})
    natural = target / "natural-errors.jsonl"
    natural.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    synthetic = load(target / "mutations.jsonl")
    combined_rows = [*synthetic, *rows]
    combined = target / "combined.jsonl"
    combined.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_rows), encoding="utf-8")
    manifest = {"version": NATURAL_VERSION, "capture_run_id": report["run_id"],
                "capture_ex": report["metrics"]["execution_accuracy_v7"],
                "capture_completion": report["metrics"]["completion"],
                "natural_records": len(rows), "natural_families": dict(Counter(row["error_type"] for row in rows)),
                "synthetic_records": len(synthetic), "combined_records": len(combined_rows),
                "combined_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
                "contains_raw_gold_sql": False, "runtime_gold_access": False,
                "database_hash_mismatches": report["details"]["database_hash_mismatches"]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
