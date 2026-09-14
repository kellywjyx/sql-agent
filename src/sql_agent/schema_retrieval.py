"""CPU hybrid retrieval over untrusted SQLite schema metadata."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter

import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_SHA256 = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"


def tokens(value: str) -> list[str]:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("_", " ")
    return [part[:-1] if part.endswith("s") and len(part) > 3 else part
            for part in re.findall(r"[a-z0-9]+", value.casefold())]


def _description_rows(database: Path) -> dict[tuple[str, str], dict]:
    folder = database.parent / "database_description"
    output = {}
    if not folder.is_dir():
        return output
    for path in sorted(folder.glob("*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    original = (row.get("original_column_name") or "").strip()
                    if original:
                        output[(path.stem.casefold(), original.casefold())] = {
                            "display_name": (row.get("column_name") or "").strip(),
                            "description": (row.get("column_description") or "").strip(),
                            "value_description": (row.get("value_description") or "").strip()[:500],
                        }
        except (OSError, UnicodeError, csv.Error):
            continue
    return output


def schema_documents(database: Path, schema: list[dict]) -> list[dict]:
    descriptions = _description_rows(database)
    documents = []
    for table in schema:
        table_name = table["table"]
        neighbours = sorted(set(table.get("references", [])))
        documents.append({"id": f"table:{table_name}", "kind": "table", "table": table_name,
                          "column": None, "text": f"table {table_name}; columns {' '.join(table['columns'])}; related tables {' '.join(neighbours)}"})
        foreign_from = {key["from"]: key for key in table.get("foreign_keys", [])}
        for column in table.get("column_details", []):
            extra = descriptions.get((table_name.casefold(), column["name"].casefold()), {})
            roles = []
            if column.get("primary_key"):
                roles.append("primary key")
            if column["name"] in foreign_from:
                key = foreign_from[column["name"]]
                target = f"{key['table']}.{key['to']}" if key.get("to") else key["table"]
                roles.append(f"foreign key to {target}")
            text = "; ".join(part for part in [
                f"column {table_name}.{column['name']}", f"type {column.get('type') or 'unknown'}",
                " ".join(roles), extra.get("display_name", ""), extra.get("description", ""),
                extra.get("value_description", "")
            ] if part)
            documents.append({"id": f"column:{table_name}.{column['name']}", "kind": "column",
                              "table": table_name, "column": column["name"], "text": text,
                              "description": extra.get("description", "")})
    return documents


def schema_identity(schema: list[dict], documents: list[dict]) -> str:
    basis = {"schema": schema, "documents": documents, "model": MODEL_NAME, "revision": MODEL_REVISION}
    return hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _rrf(rankings: list[list[str]]) -> dict[str, float]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, identifier in enumerate(dict.fromkeys(ranking)):
            scores[identifier] += 1 / (61 + rank)
    return dict(scores)


class HybridSchemaRetriever:
    def __init__(self, artifacts: Path, encoder=None):
        self.artifacts = artifacts.resolve()
        self.model_path = self.artifacts / "models/bge-small"
        self.index_root = self.artifacts / "sql-agent/schema-index"
        self._encoder = encoder

    def _model(self):
        if self._encoder is not None:
            return self._encoder
        manifest = self.model_path / "asset-manifest.json"
        if not (self.model_path / "config.json").is_file() or not manifest.is_file():
            raise RuntimeError("BGE schema model missing. Run `sql-agent index-schema --artifacts <artifacts> --database <database>` after installing the semantic dependencies.")
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("revision") != MODEL_REVISION or value.get("files", {}).get("model.safetensors") != MODEL_SHA256:
            raise RuntimeError("BGE schema model revision or checksum differs from the tested release")
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        self._encoder = SentenceTransformer(str(self.model_path), device="cpu", local_files_only=True)
        self._encoder.max_seq_length = 512
        return self._encoder

    def _encode(self, values: list[str]) -> np.ndarray:
        model = self._model()
        if callable(model) and not hasattr(model, "encode"):
            array = model(values)
        else:
            array = model.encode(values, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        array = np.asarray(array, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    def build(self, database: Path, schema: list[dict], *, force: bool = False) -> dict:
        documents = schema_documents(database, schema)
        identity = schema_identity(schema, documents)
        folder = self.index_root / identity
        manifest = folder / "manifest.json"
        if manifest.is_file() and not force:
            return json.loads(manifest.read_text(encoding="utf-8"))
        started = perf_counter()
        vectors = self._encode([document["text"] for document in documents])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "catalog.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
        with (folder / "vectors.npy").open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        result = {"schema_identity": identity, "database_name": database.name, "documents": len(documents),
                  "dimensions": int(vectors.shape[1]), "model": MODEL_NAME, "model_revision": MODEL_REVISION,
                  "model_sha256": MODEL_SHA256, "build_seconds": perf_counter() - started}
        manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def check(self, database: Path, schema: list[dict]) -> dict:
        self._model()
        identity, documents, vectors = self._load(database, schema)
        return {"schema_index": "ready", "schema_identity": identity,
                "schema_documents": len(documents), "embedding_dimensions": int(vectors.shape[1]),
                "embedding_model": MODEL_NAME, "embedding_revision": MODEL_REVISION}

    def _load(self, database: Path, schema: list[dict]):
        documents = schema_documents(database, schema)
        identity = schema_identity(schema, documents)
        folder = self.index_root / identity
        if not (folder / "manifest.json").is_file():
            raise RuntimeError(f"Schema index missing for {database.name}. Run `sql-agent index-schema --artifacts {self.artifacts} --database {database}`.")
        stored = json.loads((folder / "catalog.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy", allow_pickle=False)
        if len(stored) != len(vectors) or [row["id"] for row in stored] != [row["id"] for row in documents]:
            raise RuntimeError("Schema index catalog does not match the current database schema")
        return identity, stored, vectors

    def select(self, database: Path, question: str, schema: list[dict], *, byte_budget: int = 10000) -> tuple[str, dict]:
        started = perf_counter()
        identity, documents, vectors = self._load(database, schema)
        query_vector = self._encode(["Represent this question for retrieving relevant database columns: " + question])[0]
        dense_order = [documents[i]["id"] for i in np.argsort(-(vectors @ query_vector))]
        query_tokens = set(tokens(question))
        lexical = sorted(documents, key=lambda row: (-len(query_tokens & set(tokens(row["text"]))), row["id"]))
        lexical_order = [row["id"] for row in lexical if query_tokens & set(tokens(row["text"]))]
        normalized_question = " " + " ".join(tokens(question)) + " "
        exact_order = [row["id"] for row in documents
                       if (row.get("column") or row.get("table")) and
                       (" " + " ".join(tokens(row.get("column") or row["table"])) + " ") in normalized_question]
        description_order = [row["id"] for row in documents if row.get("description") and
                             query_tokens & set(tokens(row["description"]))]
        description_order.sort(key=lambda identifier: (
            -len(query_tokens & set(tokens(next(row["description"] for row in documents if row["id"] == identifier)))),
            identifier))
        scores = _rrf([exact_order, dense_order, lexical_order, description_order])
        by_id = {row["id"]: row for row in documents}
        ranked = sorted(documents, key=lambda row: (-scores.get(row["id"], 0), row["id"]))
        all_columns = [row for row in ranked if row["kind"] == "column"]
        if len(all_columns) <= 40:
            selected_columns = all_columns
        else:
            exact_columns = [by_id[identifier] for identifier in exact_order if by_id[identifier]["kind"] == "column"]
            selected_columns = list({row["id"]: row for row in [*exact_columns, *all_columns]}.values())[:32]
        table_seeds = [row["table"] for row in ranked if row["kind"] == "table"][:5]
        selected_tables = list(dict.fromkeys([*table_seeds, *(row["table"] for row in selected_columns)]))
        graph = {table["table"]: set(table.get("references", [])) for table in schema}
        for table in schema:
            for neighbour in table.get("references", []):
                graph.setdefault(neighbour, set()).add(table["table"])
        join_paths = []
        seeds = selected_tables
        names = set(seeds)
        for start in seeds:
            queue, visited = deque([(start, [start])]), {start}
            while queue:
                node, path = queue.popleft()
                if node in seeds and node != start:
                    names.update(path)
                    join_paths.append(path)
                for neighbour in sorted(graph.get(node, ())):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append((neighbour, [*path, neighbour]))
        by_table = {table["table"]: table for table in schema}
        chosen = defaultdict(set)
        for row in selected_columns:
            chosen[row["table"]].add(row["column"])
        retained_keys = []
        for name in list(names):
            table = by_table.get(name)
            if not table:
                continue
            for column in table.get("column_details", []):
                if column.get("primary_key"):
                    chosen[name].add(column["name"]); retained_keys.append(f"{name}.{column['name']}")
            for key in table.get("foreign_keys", []):
                if key["table"] in names:
                    chosen[name].add(key["from"])
                    retained_keys.append(f"{name}.{key['from']}")
                    if key.get("to"):
                        chosen[key["table"]].add(key["to"])
                        retained_keys.append(f"{key['table']}.{key['to']}")
        def quote(value):
            return '"' + value.replace('"', '""') + '"'
        statements = []
        ordered_tables = sorted(names, key=lambda name: min((i for i, row in enumerate(ranked) if row["table"] == name), default=9999))
        for name in ordered_tables:
            table = by_table[name]
            ordered = sorted(table["column_details"], key=lambda col: (-scores.get(f"column:{name}.{col['name']}", 0), col["name"]))
            keep = chosen[name] if len(all_columns) > 40 else {column["name"] for column in ordered}
            columns = [f"{quote(column['name'])} {column.get('type') or ''}".rstrip() for column in ordered if column["name"] in keep]
            columns += [f"FOREIGN KEY ({quote(key['from'])}) REFERENCES {quote(key['table'])}"
                        + (f" ({quote(key['to'])})" if key.get("to") else "")
                        for key in table.get("foreign_keys", []) if key["table"] in names and key["from"] in keep]
            statements.append(f"CREATE TABLE {quote(name)} (" + ", ".join(columns) + ");")
        prompt = "\n".join(statements)
        if len(prompt.encode()) > byte_budget:
            raise ValueError("Hybrid schema selection exceeds the 10 KB prompt budget")
        selected = [{"table": row["table"], "column": row["column"], "score": scores.get(row["id"], 0),
                     "exact_match": row["id"] in exact_order, "description": row.get("description", ""),
                     "retrieval_reasons": [reason for reason, present in [
                         ("exact_identifier", row["id"] in exact_order),
                         ("description_match", row["id"] in description_order),
                         ("token_overlap", row["id"] in lexical_order),
                         ("dense_similarity", row["id"] in dense_order)] if present]}
                    for row in selected_columns]
        details = {"mode": "hybrid", "schema_identity": identity, "embedding_model": MODEL_NAME,
                   "embedding_revision": MODEL_REVISION, "selected_tables": ordered_tables,
                   "selected_columns": selected, "retained_keys": sorted(set(retained_keys)),
                   "join_paths": join_paths, "prompt_bytes": len(prompt.encode()),
                   "retrieval_seconds": perf_counter() - started}
        return prompt, details
