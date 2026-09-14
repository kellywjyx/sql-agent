"""CPU-built, schema-independent correction memory from BIRD training SQL."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import sqlglot
from llm_evals import load_cases
from sqlglot import exp

from .evaluation import equivalent
from .guardrails import authorizer, validate
from .m_schema import split_question_evidence
from .v10_signature import build_signature, canonical_identity


MEMORY_VERSION = "sql-v10-correction-memory-v1"


def _execute_readonly(database: Path, sql: str, *, timeout: float = .5) -> dict:
    """Fast offline scorer with the same read-only trust boundary."""
    sql = validate(sql)
    deadline = time.monotonic() + timeout
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.enable_load_extension(False)
        connection.set_authorizer(authorizer)
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        try:
            cursor = connection.execute(sql)
            columns = [row[0] for row in cursor.description or []]
            if len(columns) > 64:
                raise RuntimeError("mutation result exceeds 64 columns")
            rows = cursor.fetchmany(10001)
            if len(rows) > 10000:
                raise RuntimeError("mutation result exceeds 10000 rows")
            return {"columns": columns, "rows": [list(row) for row in rows], "row_count": len(rows)}
        finally:
            connection.set_progress_handler(None, 0)


def _replace_aggregate(tree: exp.Expression) -> tuple[str, str] | None:
    node = next(tree.find_all(exp.AggFunc), None)
    if node:
        replacements = {"count": exp.Sum, "sum": exp.Avg, "avg": exp.Count,
                        "min": exp.Max, "max": exp.Min}
        factory = replacements.get(node.key)
        if not factory or node.this is None:
            return None
        correct = node.key
        node.replace(factory(this=node.this.copy()))
        return f"replace_{factory.__name__.casefold()}_with_{correct}", "operation"
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select and select.expressions:
        first = select.expressions[0]
        if next(first.find_all(exp.Column), None) and not next(first.find_all(exp.AggFunc), None):
            select.expressions[0].replace(exp.Count(this=first.copy()))
            return "remove_unrequested_aggregate", "operation"
    return None


def _remove_group(tree: exp.Expression) -> tuple[str, str] | None:
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select and select.args.get("group"):
        select.set("group", None)
        return "restore_grouping_grain", "grain"
    return None


def _change_predicate(tree: exp.Expression) -> tuple[str, str] | None:
    replacements = {exp.EQ: exp.NEQ, exp.NEQ: exp.EQ, exp.GT: exp.LTE,
                    exp.GTE: exp.LT, exp.LT: exp.GTE, exp.LTE: exp.GT}
    node = next((item for item in tree.walk() if type(item) in replacements), None)
    if node:
        original = node.key
        node.replace(replacements[type(node)](this=node.this.copy(), expression=node.expression.copy()))
        return f"restore_predicate_operator_{original}", "predicate"
    return None


def _remove_join(tree: exp.Expression) -> tuple[str, str] | None:
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    joins = list(select.args.get("joins") or []) if select else []
    if joins:
        # Preserve a valid query while deterministically making the relationship
        # wrong. The always-false ON clause is a clean join-local hard negative.
        joins[0].set("on", exp.EQ(this=exp.Literal.number(1), expression=exp.Literal.number(0)))
        return "restore_required_join", "join"
    return None


MUTATORS = [_replace_aggregate, _remove_group, _change_predicate, _remove_join]


def build(root: Path, *, limit: int = 500, source_case_limit: int = 150, force: bool = False) -> dict:
    root = root.resolve(); target = root / "v10/sql-agent/error-memory"
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = [root / "v3/sql-agent/data/development.jsonl", root / "v3/sql-agent/data/final.jsonl",
               root / "v5/sql-agent/data/final.jsonl"]
    all_cases = [case for path in sources for case in load_cases(path)]
    # A bounded, deterministic first release. The memory can grow in later
    # versions without changing the frozen V10 pilot.
    cases = all_cases[:source_case_limit]
    before = {str(Path(case.context["database"]).resolve()): hashlib.sha256(
              Path(case.context["database"]).read_bytes()).hexdigest() for case in cases}
    rows = []
    for case in cases:
        if len(rows) >= limit:
            break
        database = Path(case.context["database"])
        try:
            gold_sql = validate(case.expected["sql"])
            gold = _execute_readonly(database, gold_sql)
            parsed = sqlglot.parse_one(gold_sql, read="sqlite")
        except (ValueError, RuntimeError, TimeoutError, sqlite3.DatabaseError, sqlglot.errors.SqlglotError):
            continue
        natural, _ = split_question_evidence(case.input)
        for mutator in MUTATORS:
            tree = parsed.copy(); label = mutator(tree)
            if not label:
                continue
            patch, family = label; mutated_sql = tree.sql(dialect="sqlite")
            try:
                mutated = _execute_readonly(database, mutated_sql)
            except (ValueError, RuntimeError, TimeoutError, sqlite3.DatabaseError):
                continue
            if equivalent(gold, mutated, case.expected.get("ordered", False)):
                continue
            signature = build_signature(natural, [], mutated_sql, mutated)
            record = {"version": MEMORY_VERSION,
                      "identity": canonical_identity(signature, family, patch),
                      "source_case_hash": hashlib.sha256(case.id.encode()).hexdigest(),
                      "error_type": family, "minimal_patch": patch,
                      "signature": signature.model_dump(), "execution_fixed": True,
                      "gold_sql_supplied_at_runtime": False,
                      "candidate_sql_sha256": hashlib.sha256(mutated_sql.encode()).hexdigest(),
                      "correct_sql_sha256": hashlib.sha256(gold_sql.encode()).hexdigest()}
            rows.append(record)
            if len(rows) >= limit:
                break
    changed = [path for path, digest in before.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError("BIRD database changed while building V10 correction memory")
    target.mkdir(parents=True, exist_ok=True)
    output = target / "mutations.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {"version": MEMORY_VERSION, "records": len(rows),
                "families": dict(Counter(row["error_type"] for row in rows)),
                "source_cases": len(cases), "available_source_cases": len(all_cases),
                "memory_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "database_hash_mismatches": changed, "contains_raw_gold_sql": False,
                "runtime_gold_access": False}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
