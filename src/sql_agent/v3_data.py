"""Freeze internal BIRD training holdouts without consulting answers for selection."""
import json
import random
from llm_evals import EvalCase, load_cases
from llm_evals.campaign import freeze_dataset, immutable_json
from llm_evals.splits import text_groups


def prepare(root):
    previous = root / "v2/sql-agent/bird"
    target = root / "v3/sql-agent/data"
    records = json.loads(next((previous / "train").rglob("train.json")).read_text(encoding="utf-8"))
    manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    seen = set(manifest["development_train_indices"])
    seen_databases = {records[i]["db_id"] for i in seen}
    groups = text_groups([(str(i), r["question"]) for i, r in enumerate(records)])
    excluded = {groups[str(i)] for i in seen}
    used = set(excluded)
    rng = random.Random(20260829)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    datasets = {}
    for split, count in [("development", 100), ("final", 200)]:
        selected = []
        for index in indices:
            record = records[index]
            if (record["db_id"] in seen_databases) != (split == "development") or groups[str(index)] in used:
                continue
            database = next((previous / "train").rglob(record["db_id"] + ".sqlite")).resolve()
            question = record["question"] + ("\nBenchmark evidence: " + record["evidence"] if record.get("evidence") else "")
            selected.append(EvalCase(id=f"bird-train-v3-{index}", input=question,
                expected={"sql": record["SQL"], "ordered": "order by" in record["SQL"].lower()},
                context={"database": str(database)}, metadata={"upstream_index": index, "upstream_question_id": record.get("question_id", index),
                "db_id": record["db_id"], "group_id": groups[str(index)], "bootstrap_group": record["db_id"],
                "scope": "internal BIRD training holdout", "source_sha256": manifest["source"]["train"]["sha256"]}))
            used.add(groups[str(index)])
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError("Insufficient unseen BIRD groups")
        datasets[split] = freeze_dataset(target / f"{split}.jsonl", selected, "development" if split == "development" else "locked_holdout")
    immutable_json(target / "provenance.json", {"seed": 20260829, "source": manifest["source"]["train"],
        "scorer": manifest["scorer"], "v2_inspected_databases": sorted(seen_databases), "splits": datasets,
        "deduplication": "text_groups-v1; final databases disjoint from v2 and v3 development"})
