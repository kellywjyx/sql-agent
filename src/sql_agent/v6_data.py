"""Freeze V6 SQL development and untouched BIRD-development complement."""
from __future__ import annotations

import json
from pathlib import Path

from llm_evals import EvalCase, load_cases
from llm_evals.campaign import freeze_dataset, immutable_json
from llm_evals.splits import text_groups


DATA_VERSION = "sql-v6-data-v1"


def _round_robin(cases, count):
    groups = {}
    for case in cases:
        groups.setdefault(case.metadata["db_id"], []).append(case)
    for values in groups.values():
        values.sort(key=lambda case: case.id)
    selected, rows = [], [groups[key] for key in sorted(groups)]
    while len(selected) < count and any(rows):
        for row in rows:
            if row and len(selected) < count:
                selected.append(row.pop(0))
    if len(selected) != count:
        raise ValueError("Insufficient database-diverse development cases")
    return selected


def prepare(root: Path) -> dict:
    target = root / "v6/sql-agent/data"
    if (target / "final.manifest.json").exists():
        return json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    bird = root / "v2/sql-agent/bird"
    inspected = load_cases(bird / "final.jsonl")
    development = []
    for case in _round_robin(inspected, 200):
        metadata = {**case.metadata, "group_id": str(case.metadata.get("upstream_question_id", case.id)),
                    "bootstrap_group": case.metadata["db_id"], "scope": "inspected Mini-Dev development selection"}
        development.append(case.model_copy(update={"metadata": metadata}))
    development_manifest = freeze_dataset(target / "development.jsonl", development, "development")

    dev_path = bird / "dev/dev_20240627/dev.json"
    records = json.loads(dev_path.read_text(encoding="utf-8"))
    inspected_ids = {int(case.metadata["upstream_question_id"]) for case in inspected}
    group_map = text_groups([(str(index), row["question"]) for index, row in enumerate(records)])
    kept_groups = set()
    final = []
    database_root = bird / "dev/dev_20240627/dev_databases/dev_databases"
    for index, row in enumerate(records):
        group = group_map[str(index)]
        upstream_id = int(row.get("question_id", index))
        if upstream_id in inspected_ids or group in kept_groups:
            continue
        kept_groups.add(group)
        database = (database_root / row["db_id"] / f"{row['db_id']}.sqlite").resolve(strict=True)
        question = row["question"] + ("\nBenchmark evidence: " + row["evidence"] if row.get("evidence") else "")
        final.append(EvalCase(id=f"bird-dev-v6-{index}", input=question,
            expected={"sql": row["SQL"], "ordered": "order by" in row["SQL"].casefold()},
            context={"database": str(database)}, metadata={"upstream_index": index,
            "upstream_question_id": row.get("question_id", index), "db_id": row["db_id"],
            "difficulty": row.get("difficulty", "unavailable"), "group_id": group,
            "bootstrap_group": row["db_id"], "scope": "untouched BIRD development complement"}))
    if len(final) != 1033:
        raise ValueError(f"Expected 1033 identity-isolated unique final groups, found {len(final)}")
    final_manifest = freeze_dataset(target / "final.jsonl", final, "locked_holdout",
                                    excluded_groups={case.metadata["group_id"] for case in development})
    source_manifest = json.loads((bird / "manifest.json").read_text(encoding="utf-8"))
    provenance = {"version": DATA_VERSION, "source": source_manifest.get("source"),
        "scorer": source_manifest.get("scorer"), "development": development_manifest, "final": final_manifest,
        "roles": {"pilot": "first 30 round-robin development cases", "development": "200 inspected Mini-Dev cases",
                  "regression": "V5 200 cases and remaining inspected Mini-Dev cases",
                  "final": "1033 unique untouched BIRD-development complement groups"},
        "identity_correction": "Upstream question IDs exclude exactly 500 Mini-Dev cases; the remaining 1034 records contain one duplicate text group.",
        "limitation": "Development and final share BIRD development schemas; question groups are isolated."}
    immutable_json(target / "provenance.json", provenance)
    return provenance
