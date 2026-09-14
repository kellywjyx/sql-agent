"""Freeze V7 exposed development roles without opening the V6 locked final."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from llm_evals import load_cases
from llm_evals.campaign import freeze_dataset, immutable_json, open_dataset


DATA_VERSION = "sql-v7-data-v1"
PROTOCOL_VERSION = "sql-v7-protocol-v1"
SEED = 20260907


def _partition(cases):
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(str(case.metadata.get("group_id", case.id)), []).append(case)
    if any(len(rows) != 1 for rows in groups.values()):
        raise ValueError("V7 exposed development partition expects isolated unique question groups")
    by_db: dict[str, list] = {}
    for rows in groups.values():
        case = rows[0]
        by_db.setdefault(str(case.metadata["db_id"]), []).append(case)
    rng = random.Random(SEED)
    for db_id, rows in by_db.items():
        rows.sort(key=lambda case: case.id)
        rng.seed(f"{SEED}:{db_id}")
        rng.shuffle(rows)
    ordered, pools = [], [by_db[key] for key in sorted(by_db)]
    while any(pools):
        for pool in pools:
            if pool:
                ordered.append(pool.pop())
    if len(ordered) != 200:
        raise ValueError(f"Expected 200 exposed V6 development cases, found {len(ordered)}")
    return ordered[:40], ordered[40:100], ordered[100:200]


def prepare(root: Path) -> dict:
    root = root.resolve()
    target = root / "v7/sql-agent/data"
    provenance_path = target / "provenance.json"
    if provenance_path.exists():
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    source = root / "v6/sql-agent/data/development.jsonl"
    cases = load_cases(source)
    reproduction, calibration, selection = _partition(cases)
    all_groups = [set(str(case.metadata.get("group_id", case.id)) for case in split)
                  for split in [reproduction, calibration, selection]]
    if any(all_groups[left] & all_groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Question group crossed V7 exposed development roles")
    def annotate(rows, role):
        return [case.model_copy(update={"metadata": {**case.metadata, "v7_role": role,
            "bootstrap_group": case.metadata["db_id"]}}) for case in rows]
    manifests = {
        "reproduction": freeze_dataset(target / "reproduction.jsonl", annotate(reproduction, "reproduction_audit"), "development"),
        "calibration": freeze_dataset(target / "calibration.jsonl", annotate(calibration, "grounding_critic_calibration"), "development"),
        "selection": freeze_dataset(target / "selection.jsonl", annotate(selection, "development_selection"), "development"),
    }
    regression_source = root / "v5/sql-agent/data/final.jsonl"
    regression = load_cases(regression_source)
    manifests["regression"] = freeze_dataset(target / "regression.jsonl",
        annotate(regression, "inspected_v5_regression"), "inspected_regression")
    source_manifest = json.loads(source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    locked_manifest_path = root / "v6/sql-agent/data/final.manifest.json"
    locked_manifest = json.loads(locked_manifest_path.read_text(encoding="utf-8"))
    protocol = {
        "version": PROTOCOL_VERSION, "seed": SEED, "api_cost_budget_usd": 0,
        "gpu_budget_seconds": 21600, "locked_final_opened": False,
        "candidate_paths": ["arctic_direct", "arctic_ir", "qwen_decomposed"],
        "success_gates": {"candidate_oracle_ex": .70, "selected_ex": .60, "completion": .97,
                          "critic_precision": .90, "interactive_p95_seconds": 20},
        "evidence_gates": {"table_recall": .98, "column_recall": .95, "literal_recall": .85},
        "stage_budgets_seconds": {"reproduction": 2520, "grounding_ir": 2880,
            "candidate_selection": 7920, "critic": 2520, "regression": 3600, "contingency": 2160},
        "final_policy": "The 1033-case V6 locked final requires a separate approved budget and frozen V7 selection.",
    }
    immutable_json(root / "v7/sql-agent/protocol.json", protocol)
    provenance = {"version": DATA_VERSION, "seed": SEED, "source": source_manifest,
        "manifests": manifests, "locked_final_identity": locked_manifest,
        "roles": {"reproduction": 40, "calibration": 60, "selection": 100, "regression": len(regression)},
        "locked_final_opened": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    immutable_json(provenance_path, provenance)
    return provenance


def open_cases(root: Path, role: str):
    if role not in {"reproduction", "calibration", "selection", "regression"}:
        raise ValueError("V7 never opens the locked final; choose an exposed role")
    return open_dataset(root.resolve() / f"v7/sql-agent/data/{role}.jsonl",
                        purpose=f"v7_{role}", ledger=root.resolve() / "v7/exposure.jsonl")
