"""Database-isolated V11 data preparation and token-coverage freezing."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import sqlglot
from llm_evals import EvalCase, load_cases
from llm_evals.campaign import freeze_dataset, immutable_json, open_dataset
from llm_evals.splits import text_groups
from sqlglot import exp

from .v11_prompt import render_case


SEED = 20260923
DATA_VERSION = "sql-v11-data-v1"
MODEL_REPOSITORY = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "dcc04bb77301e6368aa24a609c6ea496b70503e8"
COUNTS = {"train": 1200, "validation": 150, "development": 200}
DB_COUNTS = {"train": 45, "validation": 6, "development": 7}
SHAPES = ("set_or_subquery", "join", "aggregate", "ordering", "simple")
SHAPE_COUNTS = {
    "train": {"set_or_subquery": 160, "join": 600, "aggregate": 240, "ordering": 80, "simple": 120},
    "validation": {"set_or_subquery": 20, "join": 75, "aggregate": 30, "ordering": 10, "simple": 15},
    "development": {"set_or_subquery": 26, "join": 100, "aggregate": 40, "ordering": 10, "simple": 24},
}


def query_shape(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")
    if next(tree.find_all(exp.SetOperation, exp.Subquery), None):
        return "set_or_subquery"
    if next(tree.find_all(exp.Join), None):
        return "join"
    if next(tree.find_all(exp.AggFunc), None) or next(tree.find_all(exp.Group), None):
        return "aggregate"
    if next(tree.find_all(exp.Order), None) or next(tree.find_all(exp.Limit), None):
        return "ordering"
    return "simple"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _training_paths(root: Path) -> tuple[Path, Path]:
    base = root / "v2/sql-agent/bird/train/train"
    return base / "train.json", base / "train_databases/train_databases"


def _database(root: Path, database_root: Path, db_id: str) -> Path:
    path = database_root / db_id / f"{db_id}.sqlite"
    return path.resolve(strict=True)


def _excluded_fingerprints(root: Path) -> tuple[set[str], set[str]]:
    questions, sqls = set(), set()
    for version in range(3, 11):
        folder = root / f"v{version}/sql-agent/data"
        if not folder.exists():
            continue
        for path in folder.glob("*.jsonl"):
            try:
                cases = load_cases(path)
            except (ValueError, OSError):
                continue
            for case in cases:
                questions.add(re.sub(r"\W+", " ", case.input.casefold()).strip())
                expected = case.expected.get("sql") if isinstance(case.expected, dict) else None
                if expected:
                    sqls.add(re.sub(r"\s+", " ", expected.casefold()).strip())
    return questions, sqls


def _source(root: Path) -> tuple[list[dict], dict]:
    train_path, database_root = _training_paths(root)
    records = json.loads(train_path.read_text(encoding="utf-8"))
    v5 = load_cases(root / "v5/sql-agent/data/final.jsonl")
    excluded_databases = sorted({str(case.metadata["db_id"]) for case in v5})
    excluded_questions, excluded_sqls = _excluded_fingerprints(root)
    grouped = text_groups([(str(index), row["question"]) for index, row in enumerate(records)])
    exposed_groups = {grouped[str(index)] for index, row in enumerate(records)
                      if re.sub(r"\W+", " ", row["question"].casefold()).strip() in excluded_questions}
    kept, rejected = [], defaultdict(int)
    seen_questions, seen_sqls = set(), set()
    for index, row in enumerate(records):
        db_id = str(row["db_id"])
        normalized_question = re.sub(r"\W+", " ", row["question"].casefold()).strip()
        normalized_sql = re.sub(r"\s+", " ", row["SQL"].casefold()).strip()
        reason = None
        if db_id in excluded_databases:
            reason = "v5_database"
        elif grouped[str(index)] in exposed_groups or normalized_question in excluded_questions:
            reason = "exposed_question_group"
        elif normalized_sql in excluded_sqls:
            reason = "exposed_sql"
        elif normalized_question in seen_questions:
            reason = "duplicate_question"
        elif normalized_sql in seen_sqls:
            reason = "duplicate_sql"
        if reason:
            rejected[reason] += 1
            continue
        try:
            shape = query_shape(row["SQL"])
            database = _database(root, database_root, db_id)
        except (OSError, sqlglot.errors.SqlglotError):
            rejected["invalid_source"] += 1
            continue
        seen_questions.add(normalized_question); seen_sqls.add(normalized_sql)
        kept.append({"id": f"bird-train-v11-{index}", "upstream_index": index,
                     "db_id": db_id, "question": row["question"], "sql": row["SQL"],
                     "database": str(database), "shape": shape, "group_id": grouped[str(index)]})
    return kept, {"source_records": len(records), "eligible_records": len(kept),
                  "excluded_databases": excluded_databases, "rejections": dict(rejected),
                  "source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}


def _assign_databases(rows: list[dict]) -> dict[str, str]:
    names = sorted({row["db_id"] for row in rows}, key=lambda value: _hash(f"{SEED}:{value}"))
    if len(names) != sum(DB_COUNTS.values()):
        raise ValueError(f"Expected 58 eligible databases, found {len(names)}")
    assignments, offset = {}, 0
    for role in ("train", "validation", "development"):
        for name in names[offset:offset + DB_COUNTS[role]]:
            assignments[name] = role
        offset += DB_COUNTS[role]
    return assignments


def freeze_protocol(root: Path) -> dict:
    target = root / "v11/sql-agent"
    path = target / "protocol.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows, source = _source(root)
    assignments = _assign_databases(rows)
    protocol = {
        "version": "sql-v11-protocol-v1", "seed": SEED,
        "objective": "direct end-to-end grounded SQL QLoRA adaptation",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
                  "license": "Apache-2.0", "fallback": None},
        "roles": COUNTS, "database_counts": DB_COUNTS, "database_assignments": assignments,
        "source": source, "max_length": 2048, "minimum_token_coverage": .85,
        "recipes": {"A": {"learning_rate": 2e-4}, "B": {"learning_rate": 1e-4}},
        "training": {"quantization": "NF4 double quantization", "rank": 8, "alpha": 16,
                     "dropout": .05, "batch_size": 1, "gradient_accumulation": 16,
                     "epochs": 1, "warmup_ratio": .03, "target_modules": "all-linear",
                     "optimizer": "paged_adamw_8bit", "completion_only_loss": True},
        "budget_seconds": 21600, "api_cost_budget_usd": 0,
        "locked_final_opened": False,
        "forbidden": ["3B fallback", "CPU or disk model offload", "human BIRD evidence",
                      "external API", "correction model", "multi-candidate generation", "locked final"],
    }
    immutable_json(path, protocol)
    immutable_json(target / "data/source-selection.json", {
        "version": DATA_VERSION, "source": source, "assignments": assignments,
        "eligible_by_role": {role: sum(assignments[row["db_id"]] == role for row in rows)
                             for role in COUNTS}, "locked_final_opened": False})
    return protocol


def _token_count(tokenizer, messages: list[dict], sql: str) -> int:
    value = [*messages, {"role": "assistant", "content": sql}]
    encoded = tokenizer.apply_chat_template(value, tokenize=True, add_generation_prompt=False)
    return len(encoded)


def finalize(root: Path, tokenizer, *, build_assets: bool = True) -> dict:
    root = root.resolve(); target = root / "v11/sql-agent/data"
    protocol = freeze_protocol(root)
    if (target / "provenance.json").exists():
        return json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    rows, source = _source(root); assignments = protocol["database_assignments"]
    accepted: dict[str, list[EvalCase]] = {role: [] for role in COUNTS}
    considered = defaultdict(int); overlength = defaultdict(int)
    by_role_shape: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_role_shape[(assignments[row["db_id"]], row["shape"])].append(row)
    for values in by_role_shape.values():
        values.sort(key=lambda row: _hash(f"{SEED}:{row['id']}"))
    for role, count in COUNTS.items():
        for shape in SHAPES:
            for row in by_role_shape[(role, shape)]:
                if sum(case.metadata["query_shape"] == shape for case in accepted[role]) >= SHAPE_COUNTS[role][shape]:
                    break
                considered[role] += 1
                rendered = render_case(root, Path(row["database"]), row["question"],
                                       build_assets=build_assets)
                tokens = _token_count(tokenizer, rendered["messages"], row["sql"])
                if tokens > protocol["max_length"]:
                    overlength[role] += 1
                    continue
                accepted[role].append(EvalCase(
                    id=row["id"], input=row["question"], expected={"sql": row["sql"],
                    "ordered": "order by" in row["sql"].casefold()},
                    context={"database": row["database"], "messages": rendered["messages"]},
                    metadata={"db_id": row["db_id"], "bootstrap_group": row["db_id"],
                              "group_id": row["group_id"], "query_shape": shape,
                              "token_count": tokens, "schema_mode": rendered["schema_selection"]["mode"],
                              "evidence_identity": rendered["evidence_identity"], "v11_role": role}))
        if len(accepted[role]) != count:
            raise ValueError(f"Could not fill balanced {role} role: {len(accepted[role])}/{count}")
        coverage = len(accepted[role]) / max(1, considered[role])
        if coverage < protocol["minimum_token_coverage"]:
            raise ValueError(f"{role} token coverage {coverage:.3f} is below 0.85")
    manifests = {role: freeze_dataset(target / f"{role}.jsonl", cases,
                                      "training" if role == "train" else "development")
                 for role, cases in accepted.items()}
    provenance = {"version": DATA_VERSION, "model_revision": MODEL_REVISION,
                  "source": source, "manifests": manifests,
                  "shape_quotas": SHAPE_COUNTS,
                  "token_coverage": {role: len(accepted[role]) / considered[role] for role in COUNTS},
                  "overlength": dict(overlength), "human_bird_evidence_in_inputs": False,
                  "database_disjoint": len({case.metadata["db_id"] for cases in accepted.values() for case in cases})
                                       == sum(DB_COUNTS.values()),
                  "locked_final_opened": False}
    immutable_json(target / "provenance.json", provenance)
    immutable_json(root / "v11/sql-agent/protocol-final.json", {
        **protocol, "version": "sql-v11-protocol-v2", "shape_quotas": SHAPE_COUNTS,
        "tokenizer_revision": MODEL_REVISION, "frozen_data": manifests,
        "token_coverage": provenance["token_coverage"]})
    return provenance


def open_role(root: Path, role: str):
    if role not in COUNTS:
        raise ValueError(f"Unknown V11 role: {role}")
    purpose = "training" if role == "train" else "development_selection"
    return open_dataset(root / f"v11/sql-agent/data/{role}.jsonl", purpose=purpose,
                        ledger=root / "v11/exposure.jsonl")


def prepare_evaluation_assets(root: Path, stage: str) -> list[dict]:
    root = root.resolve()
    if stage not in {"pilot", "development", "regression"}:
        raise ValueError("V11 evaluation stage must be pilot, development, or regression")
    cases = (open_role(root, "development")[:60] if stage == "pilot" else
             open_role(root, "development") if stage == "development" else
             load_cases(root / "v5/sql-agent/data/final.jsonl"))
    completed, frozen = [], []
    for case in cases:
        database = Path(case.context["database"]).resolve()
        rendered = render_case(root, database, case.input, build_assets=True)
        completed.append({"database": database.name, "schema_mode": rendered["schema_selection"]["mode"],
                          "evidence_identity": rendered["evidence_identity"]})
        frozen.append({"id": case.id, "input": case.input, "expected": case.expected,
            "context": {"database": str(database), "messages": rendered["messages"]},
            "metadata": {**case.metadata, "v11_evaluation_stage": stage,
                         "schema_mode": rendered["schema_selection"]["mode"],
                         "evidence_identity": rendered["evidence_identity"]}})
    target = root / f"v11/sql-agent/data/eval-{stage}.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in frozen)
    if target.is_file() and target.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"Frozen V11 {stage} evaluation inputs already exist with different content")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return completed
