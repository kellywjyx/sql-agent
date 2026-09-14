# BIRD final failure analysis

Execution recovery is not answer correctness. Gold execution failures retain zero credit in the full denominator; they are not solely model errors.

| Setting | Correct | Wrong executed answer | Application failure | Gold unscorable | Recovered execution / correct |
|---|---:|---:|---:|---:|---:|
| baseline | 95 | 174 | 119 | 112 | 0 / 0 |
| linking | 79 | 181 | 128 | 112 | 0 / 0 |
| corrected | 98 | 241 | 49 | 112 | 81 / 19 |

## Reference execution failures

- function_authorizer: 8 cases; example IDs: `bird-mini-dev-83`, `bird-mini-dev-106`, `bird-mini-dev-108`, `bird-mini-dev-109`, `bird-mini-dev-111`.
- schema_column_limit: 103 cases; example IDs: `bird-mini-dev-128`, `bird-mini-dev-129`, `bird-mini-dev-130`, `bird-mini-dev-131`, `bird-mini-dev-132`.
- deadline: 1 cases; example IDs: `bird-mini-dev-340`.

Full per-case reference errors are retained in the companion JSON. The worker's existing SQLITE_LIMIT_COLUMN=64 can reject an otherwise readable wide schema with a misleading 'malformed schema / too many columns' message. This is a configured compatibility limit, not evidence that the dataset file is corrupt. CURRENT_TIMESTAMP is outside the function allowlist. These controls were not relaxed during the campaign.

Outcome categories are mutually exclusive, with gold-unscorable taking precedence. A case can also have an application error; the separate attempt-category counts retain that information. Failed attempts count individually, while outcome rows count cases.

Uncaught tokenizer exceptions can end a case before an attempt is recorded and before correction. They fail closed, but this is a recovery limitation of the frozen implementation. The companion JSON separates these exception types from ordinary rejected attempts.

Empty executed results remain successful executions. Their correctness still depends on reference execution. Queries exceeding the row/byte budget fail; evaluation results are never silently truncated. Database mismatches and deadlines were not bypassed to obtain a score.

This analysis reads the frozen verdicts; it does not rerun or repair gold SQL, model SQL, database schemas, or outputs. Final inspected examples become regression material for a future campaign.
