# sql-agent: final comparison

Dataset content identity: `5cdb12d56dfb17e3953fd34f39b3f1a4d03d92d29ad334608b9b18fd4558462a`.

Cases per setting: 500. Execution gate passed: **False**.

| Setting | guarded_bird_ex | Failed / skipped | p95 seconds |
|---|---:|---:|---:|
| baseline | 0.1900 | 228 / 0 | 4.20 |
| linking | 0.1580 | 237 / 0 | 4.79 |
| corrected | 0.1960 | 156 / 0 | 9.22 |

## Paired deltas against baseline

| Candidate | Metric | Delta | 95% interval |
|---|---|---:|---|
| linking | guarded_bird_ex | -0.0320 | [-0.0620, -0.0060] |
| corrected | guarded_bird_ex | +0.0060 | [-0.0220, +0.0320] |

## Run identities

- baseline: `20260828T110044.079203Z-b0a75bf1`; predictions SHA256 `55ee21a583f3659c9364549b7a342b78c98bd0d0ce7e37d53272a467ae144e8e`.
- linking: `20260828T112019.395011Z-90187ed3`; predictions SHA256 `a3c013d6d43b2dc674c4f4cb07408442aae989c3169c6eb62a4845ef5d32ee9c`.
- corrected: `20260828T114327.756620Z-2ac85047`; predictions SHA256 `de7a11a753ea5ddf82be05355c8f431288b8b8c637479254420a8a6ed1bc4daf`.

## Limitations

- Comparisons summarize the same fixed cases, not independent benchmark replications.
- Paired intervals use 300 deterministic resamples; no multiple-comparison correction.
- Failed examples remain in denominators. A descriptive comparison does not pass a failed execution gate.
- First/subsequent latency is not proof of cold/warm residency; see the separate model-residency probe.
- Guarded 500-case BIRD SQLite Mini-Dev using the pinned upstream set-of-rows comparator; not a leaderboard result.
- Local API charges: US$0. Hardware/electricity costs were not measured.

## Execution behavior

| Setting | First-attempt accuracy | First-attempt execution | Recovered execution | Correct recoveries | Timeout cases |
|---|---:|---:|---:|---:|---:|
| baseline | 0.190 | 0.544 | 0 / 222 initial failures | 0 | 0.000 |
| linking | 0.158 | 0.526 | 0 / 233 initial failures | 0 | 0.000 |
| corrected | 0.158 | 0.526 | 81 / 232 initial failures | 19 | 0.000 |

An executed query is not necessarily a correct answer. Recovery counts condition on recorded initial rejected attempts; uncaught application exceptions with no saved attempt are still included in the full case denominator. Tokenizer exceptions can bypass correction in this frozen implementation; failure diagnostics identify them separately.

## Reference compatibility

- function_authorizer: 8 reference cases.
- schema_column_limit: 103 reference cases.
- deadline: 1 reference cases.

All such reference cases retain zero credit. SQLITE_LIMIT_COLUMN=64 can prevent reading an otherwise valid wide schema; a synthetic regression test confirms this is a configured limit, not proof of database corruption. CURRENT_TIMESTAMP remains outside the function allowlist. Query deadlines, row/byte limits and all read-only enforcement remain unchanged.

## Difficulty breakdown

| Difficulty | Cases | Full schema | Linking | Corrected |
|---|---:|---:|---:|---:|
| challenging | 102 | 0.118 | 0.078 | 0.108 |
| moderate | 250 | 0.152 | 0.124 | 0.156 |
| simple | 148 | 0.304 | 0.270 | 0.324 |

## Selection and reproduction

Full schema is the headline setting selected from development outcomes before inspecting final answers or metrics. Its paired development differences were uncertain. The interactive application's existing linking/correction workflow remains experimental, not a claimed accuracy improvement.

Generation uses local llama3.1:8b with 8,192 context tokens and at most 400 generated tokens per attempt. The three settings use the same generation parameters and a 10-second worker execution deadline. Full schema/linking use one attempt; corrected uses at most three.

[Portable comparison identities and intervals](v2-comparison.json) · [Failure diagnostics](failure-analysis.json) · [Development decision](development-decision.json) · [Commands](REPRODUCE.md).


## Database preservation

After all development and final runs, SHA-256 verification confirmed unchanged bytes for all 55 benchmark databases. [After-run audit](database-byte-audit.json) · [Before-run inventory](database-byte-inventory.json).
