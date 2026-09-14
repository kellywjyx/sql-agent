"""Freeze V10 roles before correction implementation or new inference."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from llm_evals import load_cases
from llm_evals.campaign import freeze_dataset, immutable_json, open_dataset


PROTOCOL_VERSION = "sql-v10-protocol-v1"
SEED = 20260917


def _captured_ids(root: Path) -> set[str]:
    identifiers: set[str] = set()
    for version in ("v7", "v8", "v9"):
        base = root / version / "sql-agent/eval"
        if not base.exists():
            continue
        for path in base.rglob("predictions.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    identifiers.add(str(json.loads(line)["id"]))
                except (KeyError, json.JSONDecodeError):
                    pass
    return identifiers


def prepare(root: Path) -> dict:
    root = root.resolve(); target = root / "v10/sql-agent"
    provenance_path = target / "data/provenance.json"
    if provenance_path.exists():
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    candidates = load_cases(root / "v7/sql-agent/data/calibration.jsonl")
    captured = _captured_ids(root)
    available = [case for case in candidates if case.id not in captured]
    if len(available) < 40:
        raise ValueError(f"V10 requires 40 cases without V7-V9 prediction captures; found {len(available)}")
    by_db: dict[str, list] = {}
    for case in available:
        by_db.setdefault(str(case.metadata["db_id"]), []).append(case)
    rng = random.Random(SEED)
    for db_id, rows in by_db.items():
        rows.sort(key=lambda case: case.id); rng.seed(f"{SEED}:{db_id}"); rng.shuffle(rows)
    selected = []
    pools = [by_db[key] for key in sorted(by_db)]
    while len(selected) < 40 and any(pools):
        for pool in pools:
            if pool and len(selected) < 40:
                selected.append(pool.pop())
    selected = [case.model_copy(update={"metadata": {**case.metadata,
        "v10_role": "fresh_exposed_pilot", "bootstrap_group": case.metadata["db_id"]}}) for case in selected]
    manifest = freeze_dataset(target / "data/pilot.jsonl", selected, "development")
    ids = [case.id for case in selected]
    immutable_json(target / "data/pilot-selection.json", {
        "version": "sql-v10-pilot-selection-v1", "seed": SEED,
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "ids": ids, "databases": sorted({case.metadata["db_id"] for case in selected}),
        "excluded_prior_prediction_ids": len(captured), "outcomes_inspected_at_freeze": False,
        "source": "V7 calibration role; absent from all V7-V9 prediction captures",
        "locked_final_opened": False})
    protocol = {
        "version": PROTOCOL_VERSION, "seed": SEED,
        "objective": "selective high-precision correction of four semantic error families",
        "generator": "frozen V9 scoped evidence plus Arctic Q4 single candidate",
        "families": ["operation", "grain", "predicate", "join"],
        "conditions": ["v10_a_scoped_control", "v10_b_deterministic", "v10_c_selective_qwen"],
        "runtime_candidate_limit": 1, "max_correction_calls": 1,
        "model_job_budget_seconds": 3600, "api_cost_budget_usd": 0,
        "detector": {"minimum_precision": .80, "optimize_for": "precision", "abstention_default": True},
        "pilot_gate": {"ex_lift": .075, "net_corrections": 3, "correct_query_regressions_max": 1,
                       "detection_precision": .80, "repair_success": .50, "completion": .95,
                       "p95_seconds_max": 20, "database_mutations": 0},
        "qwen_activation": "only if deterministic lift misses +0.075, calibrated detector precision >=0.80, and unpatched flagged errors remain",
        "forbidden": ["probes", "multi-candidate generation", "Talon", "RL", "fine-tuning",
                      "external API", "gold SQL or result in runtime", "locked final"],
        "locked_final_opened": False,
    }
    immutable_json(target / "protocol.json", protocol)
    memory_sources = [root / "v3/sql-agent/data/development.jsonl",
                      root / "v3/sql-agent/data/final.jsonl", root / "v5/sql-agent/data/final.jsonl"]
    provenance = {"version": "sql-v10-data-v1", "pilot": manifest,
                  "memory_sources": [{"path_role": path.parent.parent.parent.name + "/" + path.name,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "cases": len(load_cases(path))} for path in memory_sources],
                  "locked_final_opened": False}
    immutable_json(provenance_path, provenance)
    return provenance


def open_pilot(root: Path):
    root = root.resolve()
    return open_dataset(root / "v10/sql-agent/data/pilot.jsonl", purpose="v10_fresh_exposed_pilot",
                        ledger=root / "v10/exposure.jsonl")


def prepare_memory_capture(root: Path) -> dict:
    root = root.resolve(); target = root / "v10/sql-agent/data/memory-capture.jsonl"
    if target.exists():
        return json.loads(target.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    cases = load_cases(root / "v3/sql-agent/data/development.jsonl")
    by_db: dict[str, list] = {}
    for case in cases:
        by_db.setdefault(str(case.metadata["db_id"]), []).append(case)
    rng = random.Random(SEED + 1)
    for name, rows in by_db.items():
        rows.sort(key=lambda case: case.id); rng.seed(f"{SEED + 1}:{name}"); rng.shuffle(rows)
    selected, pools = [], [by_db[key] for key in sorted(by_db)]
    while len(selected) < 30 and any(pools):
        for pool in pools:
            if pool and len(selected) < 30:
                selected.append(pool.pop())
    selected = [case.model_copy(update={"metadata": {**case.metadata, "v10_role": "training_error_memory"}})
                for case in selected]
    manifest = freeze_dataset(target, selected, "development")
    immutable_json(root / "v10/sql-agent/data/memory-capture-selection.json", {
        "version": "sql-v10-memory-capture-selection-v1", "seed": SEED + 1,
        "case_count": len(selected), "ids_sha256": hashlib.sha256(
            "\n".join(case.id for case in selected).encode()).hexdigest(),
        "source": "BIRD training cases from inspected V3 development role",
        "purpose": "natural Arctic scoped-error memory; never a V10 quality gate",
        "locked_final_opened": False})
    return manifest


def open_memory_capture(root: Path):
    root = root.resolve()
    return open_dataset(root / "v10/sql-agent/data/memory-capture.jsonl",
                        purpose="v10_training_error_memory", ledger=root / "v10/exposure.jsonl")
