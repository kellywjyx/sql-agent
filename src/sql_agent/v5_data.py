"""Freeze a new internal BIRD holdout from databases absent from V3."""
from __future__ import annotations

import json
import random
from pathlib import Path

from llm_evals import EvalCase, load_cases
from llm_evals.campaign import freeze_dataset, immutable_json
from llm_evals.splits import text_groups


def prepare(root: Path):
    target = root / "v5/sql-agent/data"
    if (target / "final.jsonl").exists():
        return json.loads((target / "final.manifest.json").read_text(encoding="utf-8"))
    prior = [*load_cases(root / "v3/sql-agent/data/development.jsonl"),
             *load_cases(root / "v3/sql-agent/data/final.jsonl")]
    used_ids = {case.metadata["upstream_index"] for case in prior}
    used_databases = {case.metadata["db_id"] for case in prior}
    source = next((root / "v2/sql-agent/bird/train").rglob("train.json"))
    records = json.loads(source.read_text(encoding="utf-8"))
    source_manifest = json.loads((root / "v2/sql-agent/bird/manifest.json").read_text(encoding="utf-8"))
    groups = text_groups([(str(index), row["question"]) for index, row in enumerate(records)])
    used_groups = {case.metadata["group_id"] for case in prior}
    available = {}
    for index, row in enumerate(records):
        if index in used_ids or row["db_id"] in used_databases or groups[str(index)] in used_groups:
            continue
        available.setdefault(row["db_id"], []).append(index)
    rng = random.Random(20260901)
    for values in available.values():
        rng.shuffle(values)
    databases = sorted(available)
    selected = []
    while len(selected) < 200 and any(available.values()):
        for db_id in databases:
            if available[db_id] and len(selected) < 200:
                selected.append(available[db_id].pop())
    if len(selected) != 200 or len({records[index]["db_id"] for index in selected}) < 8:
        raise ValueError("Insufficient database-diverse unseen BIRD examples")
    cases = []
    for index in selected:
        row = records[index]
        database = next((root / "v2/sql-agent/bird/train").rglob(row["db_id"] + ".sqlite")).resolve()
        question = row["question"] + ("\nBenchmark evidence: " + row["evidence"] if row.get("evidence") else "")
        cases.append(EvalCase(id=f"bird-train-v5-{index}", input=question,
            expected={"sql": row["SQL"], "ordered": "order by" in row["SQL"].casefold()},
            context={"database": str(database)}, metadata={"upstream_index": index,
            "upstream_question_id": row.get("question_id", index), "db_id": row["db_id"],
            "group_id": groups[str(index)], "bootstrap_group": row["db_id"],
            "scope": "internal BIRD training holdout", "source_sha256": source_manifest["source"]["train"]["sha256"]}))
    manifest = freeze_dataset(target / "final.jsonl", cases, "locked_holdout", excluded_groups=used_groups)
    immutable_json(target / "provenance.json", {"seed": 20260901, "source": source_manifest["source"]["train"],
        "scorer": source_manifest["scorer"], "excluded_v3_ids": len(used_ids),
        "excluded_v3_databases": sorted(used_databases), "selected_databases": sorted({c.metadata["db_id"] for c in cases}),
        "deduplication": "text_groups-v1; exact IDs and all V3 databases excluded", "final": manifest})
    return manifest
