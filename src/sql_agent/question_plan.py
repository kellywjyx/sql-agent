"""Deterministic, answer-free question requirements for SQL review."""
from __future__ import annotations

import re


def question_plan(question: str, schema_selection: dict, value_hints: list[dict]) -> dict:
    # Benchmark evidence is useful for value/identifier linking, but it often
    # contains explanatory words such as "by" or "sum" that are not request
    # operators. Infer query shape from the natural question only.
    natural, evidence = (question.split("Benchmark evidence:", 1) + [""])[:2]
    lowered = " " + natural.casefold().replace("_", " ") + " "
    aggregate = next((name for name, terms in {
        "count": ["how many", "number of", "count"], "sum": ["total", "sum"],
        "average": ["average", "mean"], "max": ["maximum", "highest", "most"],
        "min": ["minimum", "lowest", "least"]}.items() if any(term in lowered for term in terms)), None)
    grouping = aggregate is not None and any(term in lowered for term in [" each ", " per ", " by ", "for every", "for each"])
    ordering = any(term in lowered for term in ["highest", "lowest", "top ", "bottom ", "most ", "least ", "order", "sorted", "rank"])
    limit = bool(re.search(r"\b(top|bottom)\s+\d+\b|\b(first|last|single)\b", lowered)) or any(term in lowered for term in [" the most ", " the least ", " highest ", " lowest "])
    selected = schema_selection.get("selected_columns", [])
    output_terms = any(term in lowered for term in ["list ", "show ", "which ", "who ", "what ", "state ", "give "])
    natural_tokens = {token.rstrip("s") for token in re.findall(r"[a-z0-9]+", lowered)}
    exact = [f"{row['table']}.{row['column']}" for row in selected
             if {token.rstrip('s') for token in re.findall(r"[a-z0-9]+", row["column"].casefold().replace("_", " "))}
             and {token.rstrip('s') for token in re.findall(r"[a-z0-9]+", row["column"].casefold().replace("_", " "))} <= natural_tokens]
    semantic_outputs = [f"{row['table']}.{row['column']}" for row in selected[:8]
                        if output_terms and any(term in row["column"].casefold().replace("_", " ")
                                                for term in ["name", "code", "title", "id", "currency", "language", "product"])]
    context_lowered = (natural + " " + evidence).casefold()
    explicit_literals = {
        value.casefold() for value in [
            *re.findall(r"['\"]([^'\"\n]{1,120})['\"]", natural),
            *re.findall(r"(?:=|refers to)\s*['\"]([^'\"\n]{1,120})['\"]", evidence,
                        flags=re.IGNORECASE),
        ]
    }
    def value_is_relevant(value) -> bool:
        normalized = str(value).strip().casefold()
        if not normalized:
            return False
        if normalized in explicit_literals:
            return True
        # Short categorical codes such as F or Y must be explicitly quoted;
        # substring matching them against prose creates widespread false
        # filter requirements.
        if len(normalized) < 3:
            return False
        return normalized in context_lowered
    filter_columns = sorted({f"{hint['table']}.{hint['column']}" for hint in value_hints
                             if any(value_is_relevant(value) for value in hint.get("values", []))})
    output_candidates = list(dict.fromkeys([*exact, *semantic_outputs]))[:8]
    metric_terms = {"amount", "quantity", "count", "total", "price", "revenue", "score", "number", "average", "value"}
    required_outputs = [value for value in exact if not (aggregate and metric_terms & set(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " "))))]
    if not required_outputs and not aggregate:
        required_outputs = exact
    involved_tables = {value.split(".", 1)[0] for value in [*exact, *filter_columns]}
    quoted_literals = re.findall(r"['\"]([^'\"\n]{1,120})['\"]", natural)
    evidence_literals = re.findall(r"(?:=|refers to)\s*['\"]([^'\"\n]{1,120})['\"]", evidence,
                                   flags=re.IGNORECASE)
    expected_literals = list(dict.fromkeys([*quoted_literals, *evidence_literals]))[:10]
    comparison_operators = sorted(set(re.findall(r"(?<![<>!=])(?:<=|>=|<>|!=|=|<|>)(?!=)", evidence)))
    percentage = any(term in lowered for term in [" percentage ", " percent ", "%"])
    ratio = percentage or any(term in lowered for term in [" ratio ", " proportion ", " divided by "])
    distinct = any(term in lowered for term in [" distinct ", " unique ", " different "])
    date_granularity = next((unit for unit in ["year", "month", "day", "date"] if unit in lowered), None)
    expected_projection_count = len(required_outputs) if output_terms and required_outputs else None
    return {"candidate_output_columns": output_candidates,
            "required_output_columns": required_outputs[:8] if output_terms else [], "filter_columns": filter_columns,
            "aggregation": aggregate, "grouping": grouping, "ordering": ordering, "limit": limit,
            "join_required": len(involved_tables) > 1,
            "candidate_join_paths": schema_selection.get("join_paths", [])[:10],
            "expected_literals": expected_literals, "comparison_operators": comparison_operators,
            "distinct": distinct, "ratio": ratio, "percentage": percentage,
            "date_granularity": date_granularity, "expected_projection_count": expected_projection_count,
            "requires_numeric_cleanup": [],
            "natural_question": natural.strip(),
            "source": "deterministic question/schema analysis; no benchmark answers"}
