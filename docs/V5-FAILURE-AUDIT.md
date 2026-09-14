# V5 failure audit

Audited on 2026-09-14 by the assistant (Claude); pending owner review.

## Scope

This audit covers all 91 non-correct predictions from the promoted V5 configuration (Qwen2.5-Coder 7B, full schema) on the internal 200-case BIRD-training holdout. For each case, the question, benchmark evidence, reference SQL, generated SQL, and a sample of both result sets were compared by hand. Per-case labels are stored locally in `artifacts/v5/sql-agent/audit/audit-labels.json`. Benchmark questions, reference SQL, and result values are not reproduced here, consistent with the existing evidence-redistribution policy.

## Headline

| Measure | Value |
|---|---:|
| Strict execution accuracy (recorded) | **0.545** (109/200) |
| Official set-based comparison (duplicates and order ignored) | 0.565 |
| Audited execution accuracy | **0.615** (123/200) |

The audited figure credits only the 14 cases where the reference SQL is demonstrably wrong and the model answer is a correct reading of the question and evidence. Ambiguous questions, tie conventions, and evidence contradictions are **not** credited. The strict 0.545 remains the primary, reproducible metric; the audited number is a diagnostic.

## Outcomes

| Outcome | Cases |
|---|---:|
| Model error, reference usable | 49 |
| Reference defective, model answer defensible (credited) | 14 |
| Reference defective, model answer also wrong | 10 |
| Application failure (no answer) | 13 |
| Ambiguous question or tie convention | 3 |
| Evidence contradicts the question | 2 |

In total, 24 of the 91 failures (26%) involve a defective reference, for example:

- a movie title in the reference that differs from the question
- `OR` without parentheses
- `SUBSTR` applied to a string literal instead of a column
- a join that multiplies rows
- an unrequested column

## Model errors (49)

| Cause | Cases | Typical pattern |
|---|---:|---|
| Value and format grounding | 18 | Arithmetic, sorting, or comparison on text-stored money and counts (`'US$57,500.00'`, `'1,000,000+'`, `'0:03:30'`); literal case, abbreviation, or accent mismatches (`'WI'`, `'carmel'`, `'Åke'`) |
| Query logic | 14 | Dropped constraints; unrequested `LIMIT 1` or aggregation; integer division; `NULL`s sorting first when finding a minimum |
| Schema path | 9 | Wrong table or column, joining unrelated keys, or joins that fan out rows |
| Duplicate rows | 5 | Listing entities through joins without `DISTINCT` |
| Extra projection | 3 | Returning an additional column the question did not ask for |

Application failures (13): 9 invalid SQL that correction did not repair, usually columns referenced on the wrong alias, and 4 execution timeouts.

## What this implies

1. **The largest fixable model weakness is grounding in stored values.** The V5 prompt shows column names and types only. Descriptions and example values from BIRD's `database_description` files are not supplied.
2. **Several failures are prompt-addressable conventions:** real division, exact stored spelling, projection discipline, `DISTINCT` for repeated entities, and no unrequested `LIMIT`.
3. **Training-split references are noisy enough to understate model quality** by about 7 points on this holdout. Future claims should keep the strict metric primary and report audited results separately.

These findings define the V12 experiment ([V12.md](V12.md)).
