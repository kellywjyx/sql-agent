# BIRD development failure analysis

Execution recovery is not answer correctness. Gold execution failures retain zero credit in the full denominator; they are not solely model errors.

| Setting | Correct | Wrong executed answer | Application failure | Gold unscorable | Recovered execution / correct |
|---|---:|---:|---:|---:|---:|
| baseline | 28 | 48 | 18 | 6 | 0 / 0 |
| linking | 24 | 48 | 22 | 6 | 0 / 0 |
| corrected | 26 | 57 | 11 | 6 | 11 / 1 |

## Reference execution failures

- deadline: 2 cases; example IDs: `bird-train-0`, `bird-train-32`.
- result_budget: 1 cases; example IDs: `bird-train-31`.
- schema_reference_mismatch: 3 cases; example IDs: `bird-train-76`, `bird-train-77`, `bird-train-78`.

Full per-case reference errors are retained in the companion JSON. The worker's existing SQLITE_LIMIT_COLUMN=64 can reject an otherwise readable wide schema with a misleading 'malformed schema / too many columns' message. This is a configured compatibility limit, not evidence that the dataset file is corrupt. CURRENT_TIMESTAMP is outside the function allowlist. These controls were not relaxed during the campaign.

Outcome categories are mutually exclusive, with gold-unscorable taking precedence. A case can also have an application error; the separate attempt-category counts retain that information. Failed attempts count individually, while outcome rows count cases.

Uncaught tokenizer exceptions can end a case before an attempt is recorded and before correction. They fail closed, but this is a recovery limitation of the frozen implementation. The companion JSON separates these exception types from ordinary rejected attempts.

Empty executed results remain successful executions. Their correctness still depends on reference execution. Queries exceeding the row/byte budget fail; evaluation results are never silently truncated. Database mismatches and deadlines were not bypassed to obtain a score.

This analysis reads the frozen verdicts; it does not rerun or repair gold SQL, model SQL, database schemas, or outputs. Final inspected examples become regression material for a future campaign.
