"""Scoped evidence contracts for V9 execution-grounded reasoning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .v8_evidence import EvidenceFact, StructuredEvidencePacket


SCOPE_VERSION = "sql-v9-evidence-scope-v1"
SQL_ROLE = Literal["predicate", "identifier_binding", "conversion", "join"]
FORBIDDEN = ("projection", "aggregation", "grouping", "ordering", "limit")


class ScopedEvidenceFact(BaseModel):
    evidence_id: str
    fact: EvidenceFact
    allowed_effects: list[SQL_ROLE]
    forbidden_effects: list[str] = Field(default_factory=list)
    instruction: str
    verified: bool = False
    verification_id: str | None = None


@dataclass(frozen=True)
class ScopedEvidencePacket:
    identity: str
    facts: list[ScopedEvidenceFact]
    text: str
    byte_count: int
    source_identity: str


def _scope(fact: EvidenceFact, position: int) -> ScopedEvidenceFact:
    if fact.type in {"literal_value", "encoded_value", "normalization"}:
        allowed: list[SQL_ROLE] = ["predicate"]
        instruction = (
            f"Use only to map the question concept to a predicate on {fact.table}.{fact.column}. "
            "It does not request a result column or an aggregate."
        )
        forbidden = list(FORBIDDEN)
    elif fact.type == "numeric_or_storage":
        allowed = ["conversion"]
        instruction = (
            f"Use only to convert {fact.table}.{fact.column} when the question already requires numeric, "
            "date, or percentage comparison/arithmetic. Storage format does not imply SUM, AVG, COUNT, or GROUP BY."
        )
        forbidden = ["aggregation", "grouping", "ordering", "limit"]
    elif fact.type == "join_relationship":
        allowed = ["join"]
        instruction = "Use only as an available join predicate when both related tables are otherwise required."
        forbidden = list(FORBIDDEN)
    else:
        allowed = ["identifier_binding"]
        instruction = (
            f"Use only to bind a question concept to {fact.table}.{fact.column}. "
            "The description does not imply filtering, aggregation, grouping, ordering, or projection."
        )
        forbidden = ["predicate", *FORBIDDEN]
    stable = json.dumps(fact.model_dump(), sort_keys=True, ensure_ascii=False)
    evidence_id = f"E{position}-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:8]}"
    return ScopedEvidenceFact(evidence_id=evidence_id, fact=fact, allowed_effects=allowed,
                              forbidden_effects=forbidden, instruction=instruction)


def scope_packet(packet: StructuredEvidencePacket, *, byte_budget: int = 4096,
                 fact_limit: int = 12) -> ScopedEvidencePacket:
    scoped = [_scope(fact, index + 1) for index, fact in enumerate(packet.facts[:fact_limit])]
    kept: list[ScopedEvidenceFact] = []
    sections: list[str] = []
    for row in scoped:
        fact = row.fact
        mapping = f"{fact.table}.{fact.column}" if fact.column else fact.table
        if fact.db_value is not None:
            mapping += f" = {json.dumps(fact.db_value, ensure_ascii=False)}"
        text = (f"[{row.evidence_id}] type={fact.type}; concept={json.dumps(fact.phrase)}; "
                f"mapping={mapping}; allowed={','.join(row.allowed_effects)}; "
                f"forbidden={','.join(row.forbidden_effects)}. {row.instruction}")
        candidate = "\n".join([*sections, text])
        if len(candidate.encode("utf-8")) > byte_budget:
            break
        kept.append(row); sections.append(text)
    rendered = "\n".join(sections)
    payload = {"version": SCOPE_VERSION, "source": packet.identity,
               "facts": [row.model_dump() for row in kept]}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return ScopedEvidencePacket(identity=identity, facts=kept, text=rendered,
                                byte_count=len(rendered.encode("utf-8")), source_identity=packet.identity)
