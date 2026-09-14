"""Freeze V8 to the already exposed V7 reproduction cases only."""
from __future__ import annotations

import json
import random
from pathlib import Path

from llm_evals import load_cases
from llm_evals.campaign import freeze_dataset, immutable_json, open_dataset


PROTOCOL_VERSION = "sql-v8-protocol-v1"
SEED = 20260911


def prepare(root: Path) -> dict:
    root = root.resolve()
    target = root / "v8/sql-agent"
    provenance_path = target / "data/provenance.json"
    if provenance_path.exists():
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    source = root / "v7/sql-agent/data/reproduction.jsonl"
    cases = load_cases(source)
    if len(cases) != 40:
        raise ValueError(f"V8 expects the 40 exposed V7 reproduction cases, found {len(cases)}")
    annotated = [case.model_copy(update={"metadata": {**case.metadata,
        "v8_role": "inspected_evidence_diagnostic", "bootstrap_group": case.metadata["db_id"]}})
        for case in cases]
    # llm-evals intentionally has a small role vocabulary.  These cases are
    # development data, with their prior inspection recorded in metadata.
    manifest = freeze_dataset(target / "data/decomposition.jsonl", annotated, "development")
    v7_decision = json.loads((root / "v7/sql-agent/reproduction-decision.json").read_text(encoding="utf-8"))
    protocol = {
        "version": PROTOCOL_VERSION,
        "seed": SEED,
        "objective": "reconstruct database-grounded evidence without oracle evidence at inference",
        "source_role": "exposed V7 reproduction audit",
        "case_count": 40,
        "pilot_case_count": 20,
        "generator": "Arctic Q4 reference-like single path",
        "api_cost_budget_usd": 0,
        "locked_final_opened": False,
        "forbidden": ["oracle evidence at runtime", "Q5", "Talon", "multi-candidate generation",
                      "critic selection", "locked final"],
        "gates": {
            "taxonomy_coverage": .90,
            "recoverability_coverage": .90,
            "literal_recall": .60,
            "value_column_accuracy": .80,
            "evidence_precision": .70,
            "pilot_min_ex_lift": .075,
            "recoveries_must_exceed_regressions": True,
        },
        "controls": {
            "no_evidence_run": v7_decision["conditions"]["reference_none"]["run_id"],
            "oracle_evidence_run": v7_decision["conditions"]["reference_oracle"]["run_id"],
        },
        "model_job_budget_seconds": 3600,
        "ablation_conditions": ["generated_literals", "generated_descriptions",
                                "generated_normalization", "generated_all"],
    }
    immutable_json(target / "protocol.json", protocol)
    provenance = {"version": "sql-v8-data-v1", "manifest": manifest,
                  "v7_source": "v7/sql-agent/data/reproduction.jsonl", "locked_final_opened": False}
    immutable_json(provenance_path, provenance)
    return provenance


def open_cases(root: Path):
    root = root.resolve()
    return open_dataset(root / "v8/sql-agent/data/decomposition.jsonl", purpose="v8_exposed_decomposition",
                        ledger=root / "v8/exposure.jsonl")


def prepare_pilot(root: Path) -> dict:
    """Freeze all evidence-limited cases plus deterministic B/C controls."""
    root = root.resolve()
    path = root / "v8/sql-agent/data/pilot.jsonl"
    if path.is_file():
        return json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    cases = open_cases(root)
    groups = {row["id"]: row["group"] for row in
              (json.loads(line) for line in (root / "v8/sql-agent/decomposition.jsonl").read_text(
                  encoding="utf-8").splitlines())}
    rng = random.Random(SEED)
    group_a = [case for case in cases if groups[case.id] == "A"]
    group_b = [case for case in cases if groups[case.id] == "B"]
    group_c = [case for case in cases if groups[case.id] == "C"]
    rng.shuffle(group_b); rng.shuffle(group_c)
    selected = {case.id for case in [*group_a, *group_b[:6], *group_c[:5]]}
    pilot = [case.model_copy(update={"metadata": {**case.metadata, "v8_pilot": True}})
             for case in cases if case.id in selected]
    if len(pilot) != 20 or len(group_a) != 9:
        raise ValueError("V8 pilot requires nine evidence-limited cases and eleven controls")
    manifest = freeze_dataset(path, pilot, "development")
    immutable_json(root / "v8/sql-agent/data/pilot-selection.json", {
        "version": "sql-v8-pilot-selection-v1", "seed": SEED, "ids": [case.id for case in pilot],
        "groups": {letter: sum(groups[case.id] == letter for case in pilot) for letter in "ABCD"},
        "source": "already exposed V7 reproduction cases", "locked_final_opened": False})
    return manifest


def open_pilot(root: Path):
    root = root.resolve()
    return open_dataset(root / "v8/sql-agent/data/pilot.jsonl", purpose="v8_generated_evidence_pilot",
                        ledger=root / "v8/exposure.jsonl")
