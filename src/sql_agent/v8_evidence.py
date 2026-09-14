"""Typed evidence routers over the V8 database-side knowledge cache."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .schema_retrieval import tokens
from .guardrails import authorizer
from .value_index import normalize_value, question_phrases
from .v8_knowledge import KnowledgeCache, char_ngrams, word_tokens


EVIDENCE_VERSION = "sql-v8-structured-evidence-v1"
STOP = {"what", "which", "who", "where", "when", "please", "give", "show", "list", "find",
        "return", "have", "with", "from", "that", "this", "them", "their", "being", "all",
        "the", "and", "for", "are", "was", "were", "had", "has", "how", "many", "much",
        "of", "in", "to", "is", "a", "an", "on", "by", "at", "as"}


class EvidenceFact(BaseModel):
    phrase: str
    type: str
    table: str
    column: str | None = None
    db_value: str | None = None
    confidence: float = Field(ge=0, le=1)
    source: str
    support: dict = Field(default_factory=dict)
    transformation: str | None = None


@dataclass(frozen=True)
class StructuredEvidencePacket:
    identity: str
    mode: str
    facts: list[EvidenceFact]
    text: str
    byte_count: int
    candidates_considered: int
    probe_count: int


def natural_question(question: str) -> str:
    return question.split("Benchmark evidence:", 1)[0].strip()


def _concept_phrases(question: str) -> list[str]:
    natural = natural_question(question)
    phrases = question_phrases(natural)
    useful = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", natural)
              if word.casefold() not in STOP and len(word) > 2]
    phrases.extend(useful)
    phrases.extend(" ".join(useful[index:index + width]) for width in (2, 3)
                   for index in range(max(0, len(useful) - width + 1)))
    result = []
    for phrase in phrases:
        normalized = normalize_value(phrase)
        if normalized in STOP or not normalized or (len(normalized) < 2 and not normalized.isdigit()):
            continue
        if normalized.isdigit() and not re.search(
                rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])", natural):
            continue
        if normalized not in {normalize_value(item) for item in result}:
            result.append(phrase)
    return result[:100]


def _value_score(phrase: str, value: str) -> tuple[float, str] | None:
    left, right = normalize_value(phrase), normalize_value(value)
    if not left or not right:
        return None
    if left == right:
        return .99, "exact_normalized_value"
    if len(left) >= 3 and (left in right or right in left) and min(len(left), len(right)) / max(len(left), len(right)) >= .75:
        return .91, "contained_normalized_value"
    # Codes and numbers require an exact or tightly contained match. Token or
    # character fuzziness turns TR000 into unrelated atom IDs such as TR000_1.
    if any(character.isdigit() for character in left + right):
        return None
    left_tokens, right_tokens = set(word_tokens(left)), set(word_tokens(right))
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    if overlap >= .5:
        return .84, "value_token_overlap"
    left_grams, right_grams = set(char_ngrams(left)), set(char_ngrams(right))
    similarity = len(left_grams & right_grams) / max(1, len(left_grams | right_grams))
    if similarity >= .65:
        return .78, "value_character_similarity"
    return None


def _year_mapping(question: str, value: str) -> tuple[float, str] | None:
    if "season" not in question.casefold():
        return None
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    value_years = re.findall(r"\b(?:19|20)\d{2}\b", value)
    if years and value_years and years[0] == value_years[-1] and normalize_value(years[0]) != normalize_value(value):
        return .94, "season_end_year_mapping"
    return None


class EvidenceReconstructor:
    def __init__(self, artifacts: Path):
        self.cache = KnowledgeCache(artifacts)

    def packet(self, database: Path, question: str, schema: list[dict], *, mode: str = "all",
               byte_budget: int = 4096, fact_limit: int = 12,
               minimum_confidence: float = .70) -> StructuredEvidencePacket:
        if mode not in {"literals", "descriptions", "normalization", "all"}:
            raise ValueError("V8 evidence mode must be literals, descriptions, normalization, or all")
        manifest, path = self.cache.load(database, schema)
        natural = natural_question(question)
        phrases = _concept_phrases(natural)
        question_terms = {term for term in tokens(natural) if term not in STOP}
        candidates: list[EvidenceFact] = []
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            columns = connection.execute("SELECT * FROM columns").fetchall()
            relationships = connection.execute("SELECT * FROM relationships").fetchall()
            # Do not deserialize the complete value cache for each question.
            # Use its exact/token indexes and retain a bounded candidate pool.
            value_candidates: dict[tuple, sqlite3.Row] = {}
            for phrase in phrases:
                normalized = normalize_value(phrase)
                if not normalized:
                    continue
                matches = connection.execute(
                    "SELECT * FROM values_index WHERE normalized_value=? "
                    "OR (length(normalized_value)>=3 AND instr(?,normalized_value)>0) "
                    "OR (length(?)>=3 AND instr(normalized_value,?)>0) LIMIT 20",
                    (normalized, normalized, normalized, normalized)).fetchall()
                for value in matches:
                    value_candidates[(value["table_name"], value["column_name"], value["original_value"])] = value
            terms = sorted({term for phrase in phrases for term in word_tokens(phrase) if len(term) >= 2})[:80]
            if terms:
                placeholders = ",".join("?" for _ in terms)
                matches = connection.execute(
                    "SELECT DISTINCT v.* FROM values_index v JOIN value_terms t "
                    "ON v.table_name=t.table_name AND v.column_name=t.column_name AND v.original_value=t.original_value "
                    f"WHERE t.term_type='token' AND t.term IN ({placeholders}) LIMIT 1000", terms).fetchall()
                for value in matches:
                    value_candidates[(value["table_name"], value["column_name"], value["original_value"])] = value
            years = re.findall(r"\b(?:19|20)\d{2}\b", natural)
            for year in years:
                for value in connection.execute(
                        "SELECT * FROM values_index WHERE normalized_value LIKE ? LIMIT 100", (f"%{year}%",)).fetchall():
                    value_candidates[(value["table_name"], value["column_name"], value["original_value"])] = value
            values = list(value_candidates.values())
            categorical_values = connection.execute(
                "SELECT v.* FROM values_index v JOIN columns c "
                "ON v.table_name=c.table_name AND v.column_name=c.column_name WHERE c.categorical=1").fetchall()

        table_context = {}
        for row in columns:
            table_context.setdefault(row["table_name"], []).extend([
                row["column_name"], row["display_name"] or "", row["description"] or "",
                row["value_description"] or ""])
        column_relevance = {}
        for row in columns:
            searchable = " ".join([row["table_name"], row["column_name"], row["display_name"] or "",
                                   row["description"] or "", row["value_description"] or ""])
            overlap = question_terms & set(tokens(searchable))
            identifier_overlap = question_terms & set(tokens(row["table_name"] + " " + row["column_name"]))
            table_overlap = question_terms & set(tokens(" ".join(table_context[row["table_name"]])))
            column_relevance[(row["table_name"], row["column_name"])] = (
                len(overlap) + len(table_overlap) + 2 * len(identifier_overlap))

        # Up to four source probes cover high-cardinality values omitted from
        # the cache's representative samples. They are exact, parameterized,
        # deadline-bounded and run through the unchanged read-only authorizer.
        found_norms = {normalize_value(row["original_value"]) for row in values}
        explicit = [phrase for phrase in phrases if
                    ((normalize_value(phrase) not in found_norms) or re.fullmatch(r"[A-Za-z]+\d+", phrase)) and
                    (re.search(r"\d", phrase) or phrase[:1].isupper() or " " in phrase)]
        probes = 0
        if explicit:
            with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as source:
                source.execute("PRAGMA query_only=ON"); source.enable_load_extension(False); source.set_authorizer(authorizer)
                for phrase in explicit:
                    identifier_like = bool(re.fullmatch(r"[A-Za-z]+\d+", phrase))
                    probe_columns = sorted(columns, key=lambda row: (
                        -(column_relevance[(row["table_name"], row["column_name"])] +
                          (4 if identifier_like and "id" in row["column_name"].casefold() else 0)),
                        row["table_name"], row["column_name"]))
                    for row in probe_columns:
                        if probes >= 4:
                            break
                        if column_relevance[(row["table_name"], row["column_name"])] <= 0:
                            continue
                        probes += 1
                        deadline = time.monotonic() + 1
                        source.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
                        try:
                            table = '"' + row["table_name"].replace('"', '""') + '"'
                            column = '"' + row["column_name"].replace('"', '""') + '"'
                            match = source.execute(
                                f"SELECT {column}, COUNT(*) FROM {table} WHERE {column}=? GROUP BY {column} LIMIT 1",
                                (phrase,)).fetchone()
                        except sqlite3.DatabaseError:
                            match = None
                        finally:
                            source.set_progress_handler(None, 0)
                        if match:
                            values.append({"table_name": row["table_name"], "column_name": row["column_name"],
                                           "original_value": str(match[0]), "normalized_value": normalize_value(match[0]),
                                           "frequency": int(match[1]), "pattern": None, "numeric_value": None,
                                           "indexed_scope": "bounded_read_only_probe"})
                    if probes >= 4:
                        break
            source.close()

        if mode in {"literals", "all"}:
            for row in values:
                best = None
                for phrase in phrases:
                    scored = _value_score(phrase, row["original_value"])
                    if scored and (best is None or scored[0] > best[0]):
                        best = (*scored, phrase)
                if best:
                    candidates.append(EvidenceFact(
                        phrase=best[2], type="literal_value", table=row["table_name"],
                        column=row["column_name"], db_value=row["original_value"], confidence=best[0],
                        source="database_value", support={"match": best[1], "frequency": row["frequency"],
                                                          "scope": row["indexed_scope"],
                                                          "column_relevance": column_relevance.get(
                                                              (row["table_name"], row["column_name"]), 0)}))

        if mode in {"normalization", "all"}:
            for row in values:
                mapping = _year_mapping(natural, row["original_value"])
                if mapping:
                    candidates.append(EvidenceFact(
                        phrase=re.findall(r"\b(?:19|20)\d{2}\b", natural)[0], type="normalization",
                        table=row["table_name"], column=row["column_name"], db_value=row["original_value"],
                        confidence=mapping[0], source="database_value_pattern",
                        support={"rule": mapping[1], "column_relevance": column_relevance.get(
                            (row["table_name"], row["column_name"]), 0)}, transformation="season_end_year"))
            numeric_words = {"percent", "percentage", "average", "ratio", "rate", "total", "sum",
                             "less", "over", "more", "least", "most", "year", "date", "time"}
            for row in columns:
                description = " ".join([row["display_name"] or "", row["description"] or "",
                                        row["value_description"] or ""])
                overlap = sorted(question_terms & set(tokens(description)))
                storage = None
                if row["numeric_text_rate"] and row["numeric_text_rate"] >= .8:
                    storage = "numeric_text_cast"
                elif row["percentage_rate"] and row["percentage_rate"] >= .5:
                    storage = "printed_percentage"
                elif row["date_rate"] and row["date_rate"] >= .5:
                    storage = "date_text"
                if (overlap and question_terms & numeric_words) or storage:
                    confidence = .90 if overlap and row["value_description"] else .76
                    candidates.append(EvidenceFact(
                        phrase=" ".join(overlap) or storage or "storage", type="numeric_or_storage",
                        table=row["table_name"], column=row["column_name"], confidence=confidence,
                        source="column_profile_and_description", transformation=storage,
                        support={"description": description[:500], "matched_terms": overlap,
                                 "numeric_min": row["numeric_min"], "numeric_max": row["numeric_max"]}))

        if mode in {"descriptions", "all"}:
            values_by_column: dict[tuple[str, str], list[sqlite3.Row]] = {}
            for value in categorical_values:
                values_by_column.setdefault((value["table_name"], value["column_name"]), []).append(value)
            for row in columns:
                description = " ".join([row["display_name"] or "", row["description"] or "",
                                        row["value_description"] or ""])
                overlap = sorted(question_terms & set(tokens(description)))
                identifier_overlap = sorted(question_terms & set(tokens(row["table_name"] + " " + row["column_name"])))
                if overlap or identifier_overlap:
                    confidence = min(.95, .72 + .04 * len(overlap) + .03 * len(identifier_overlap))
                    candidates.append(EvidenceFact(
                        phrase=" ".join(overlap or identifier_overlap), type="column_semantics",
                        table=row["table_name"], column=row["column_name"], confidence=confidence,
                        source="database_description" if description else "schema_identifier",
                        support={"description": description[:700], "matched_terms": overlap,
                                 "identifier_terms": identifier_overlap}))
                # Encoded categories are a separate relation: the question may
                # contain the meaning while the database stores only the code.
                # Require the code and concept to co-occur in a source-provided
                # description segment; never infer a mapping from the code alone.
                segments = [part.strip() for part in re.split(r"[;\n\r•]+", row["value_description"] or "")
                            if part.strip()]
                for value in values_by_column.get((row["table_name"], row["column_name"]), []):
                    normalized_value = normalize_value(value["original_value"])
                    for segment in segments:
                        segment_terms = set(tokens(segment))
                        concept = sorted((question_terms & segment_terms) - set(tokens(normalized_value)) -
                                         {"common", "evidence", "refer", "mean", "indicate", "value",
                                          "table", "character", "user", "score", "attribute"})
                        normalized_segment = normalize_value(segment)
                        explicit_relation = bool(
                            re.search(rf"(^|[^a-z0-9]){re.escape(normalized_value)}\s*(?::|=|means|indicates|stands for|refers to)", normalized_segment)
                            or re.search(rf"(?::|=|means|indicates|stands for|refers to)\s*{re.escape(normalized_value)}([^a-z0-9]|$)", normalized_segment))
                        threshold_relation = bool(concept and re.search(
                            rf"\b{'|'.join(map(re.escape, concept))}\b\s*:\s*(?:[<>]=?|at least|at most)\s*{re.escape(normalized_value)}\b",
                            normalized_segment))
                        if normalized_value in normalized_segment and (threshold_relation or
                                (explicit_relation and len(concept) >= 2)):
                            candidates.append(EvidenceFact(
                                phrase=" ".join(concept), type="encoded_value", table=row["table_name"],
                                column=row["column_name"], db_value=value["original_value"], confidence=.94,
                                source="database_value_description",
                                support={"description_segment": segment[:500],
                                         "frequency": value["frequency"]},
                                transformation=(re.search(r"[<>]=?", normalized_segment).group(0)
                                                if threshold_relation and re.search(r"[<>]=?", normalized_segment) else None)))
                            break
            selected_tables = {fact.table for fact in candidates if fact.confidence >= .80}
            for relation in relationships:
                if relation["source_table"] in selected_tables or relation["target_table"] in selected_tables:
                    candidates.append(EvidenceFact(
                        phrase="foreign-key path", type="join_relationship", table=relation["source_table"],
                        column=relation["source_column"], confidence=.92, source="sqlite_foreign_key",
                        support={"target_table": relation["target_table"],
                                 "target_column": relation["target_column"]}))

        candidates = [fact for fact in candidates if fact.confidence >= minimum_confidence]
        literal_groups: dict[tuple[str, str], list[EvidenceFact]] = {}
        for fact in candidates:
            if fact.type == "literal_value":
                literal_groups.setdefault((normalize_value(fact.phrase), normalize_value(fact.db_value or "")), []).append(fact)
        suppressed = set()
        for facts in literal_groups.values():
            if len(facts) <= 1:
                continue
            best_relevance = max(int(fact.support.get("column_relevance", 0)) for fact in facts)
            if best_relevance:
                suppressed.update(id(fact) for fact in facts
                                  if int(fact.support.get("column_relevance", 0)) < best_relevance)
        candidates = [fact for fact in candidates if id(fact) not in suppressed]
        normalization_groups: dict[tuple[str, str], list[EvidenceFact]] = {}
        for fact in candidates:
            if fact.type == "normalization":
                normalization_groups.setdefault((normalize_value(fact.phrase), fact.transformation or ""), []).append(fact)
        suppressed = set()
        for facts in normalization_groups.values():
            best_relevance = max(int(fact.support.get("column_relevance", 0)) for fact in facts)
            if best_relevance:
                suppressed.update(id(fact) for fact in facts
                                  if int(fact.support.get("column_relevance", 0)) < best_relevance)
        candidates = [fact for fact in candidates if id(fact) not in suppressed]
        # Prefer source-backed values, then transformations and descriptions.  Deduplicate the
        # semantic relation, not its wording.
        priority = {"literal_value": 0, "encoded_value": 1, "normalization": 2, "numeric_or_storage": 3,
                    "column_semantics": 4, "join_relationship": 5}
        candidates.sort(key=lambda fact: (-fact.confidence, -int(fact.support.get("column_relevance", 0)),
                                          priority.get(fact.type, 9), fact.table,
                                          fact.column or "", fact.db_value or ""))
        unique, seen = [], set()
        for fact in candidates:
            key = (fact.type, fact.table.casefold(), (fact.column or "").casefold(),
                   normalize_value(fact.db_value or ""), fact.transformation)
            if key not in seen:
                seen.add(key); unique.append(fact)
        kept: list[EvidenceFact] = []
        for fact in unique:
            candidate = [*kept, fact]
            rendered = self.render(candidate)
            if len(candidate) <= fact_limit and len(rendered.encode("utf-8")) <= byte_budget:
                kept.append(fact)
        text = self.render(kept)
        basis = {"version": EVIDENCE_VERSION, "cache": manifest["identity"], "question": natural,
                 "mode": mode, "facts": [fact.model_dump() for fact in kept]}
        identity = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return StructuredEvidencePacket(identity, mode, kept, text, len(text.encode("utf-8")),
                                        len(candidates), probes)

    @staticmethod
    def render(facts: list[EvidenceFact]) -> str:
        lines = []
        for fact in facts:
            identifier = f"{fact.table}.{fact.column}" if fact.column else fact.table
            if fact.db_value is not None:
                line = f"- {fact.phrase!r} maps to {identifier} = {fact.db_value!r}."
            elif fact.type == "join_relationship":
                line = (f"- Join {identifier} to {fact.support.get('target_table')}."
                        f"{fact.support.get('target_column') or ''}")
            else:
                detail = fact.support.get("description") or fact.transformation or "schema match"
                line = f"- {fact.phrase!r} relates to {identifier}: {detail}"
            lines.append(line)
        return "\n".join(lines)
