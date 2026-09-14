# sql-agent: development comparison

Dataset content identity: `0f0b5441afc06eadd530b47486e7d92789b7cdd21feac2fd7fbb32bb4e5ba191`.

Cases per setting: 100. Execution gate passed: **False**.

| Setting | guarded_bird_ex | Failed / skipped | p95 seconds |
|---|---:|---:|---:|
| baseline | 0.2800 | 22 / 0 | 4.38 |
| linking | 0.2400 | 25 / 0 | 3.74 |
| corrected | 0.2600 | 13 / 0 | 9.30 |

## Paired deltas against baseline

| Candidate | Metric | Delta | 95% interval |
|---|---|---:|---|
| linking | guarded_bird_ex | -0.0400 | [-0.1200, +0.0300] |
| corrected | guarded_bird_ex | -0.0200 | [-0.0900, +0.0500] |

## Run identities

- baseline: `20260828T104247.167114Z-4a1e2ced`; predictions SHA256 `02fc4af18211a8a5a03a5adbd58ccc32c0cf7c3869ccd2da2e09dfb5ae7f37a4`.
- linking: `20260828T104653.239981Z-34e93d49`; predictions SHA256 `e5e430919f7f67e31c71805f246ee6c178f8cea34667235467e531c494c5ec0e`.
- corrected: `20260828T105131.213647Z-faaa4e61`; predictions SHA256 `bb90706b279d755a02995af33d4bf89bbde1e4e7f51121455b45a9b57a28e169`.

## Limitations

- Comparisons summarize the same fixed cases, not independent benchmark replications.
- Paired intervals use 300 deterministic resamples; no multiple-comparison correction.
- Failed examples remain in denominators. A descriptive comparison does not pass a failed execution gate.
- First/subsequent latency is not proof of cold/warm residency; see the separate model-residency probe.
- Guarded 100 BIRD training development cases using the pinned upstream set-of-rows comparator; not a leaderboard result.
- Local API charges: US$0. Hardware/electricity costs were not measured.

## Execution behavior

| Setting | First-attempt accuracy | First-attempt execution | Recovered execution | Correct recoveries | Timeout cases |
|---|---:|---:|---:|---:|---:|
| baseline | 0.280 | 0.780 | 0 / 22 initial failures | 0 | 0.020 |
| linking | 0.240 | 0.750 | 0 / 25 initial failures | 0 | 0.020 |
| corrected | 0.250 | 0.760 | 11 / 24 initial failures | 1 | 0.030 |

An executed query is not necessarily a correct answer. Recovery counts condition on recorded initial rejected attempts; uncaught application exceptions with no saved attempt are still included in the full case denominator. Tokenizer exceptions can bypass correction in this frozen implementation; failure diagnostics identify them separately.

## Reference compatibility

- deadline: 2 reference cases.
- result_budget: 1 reference cases.
- schema_reference_mismatch: 3 reference cases.

All such reference cases retain zero credit. SQLITE_LIMIT_COLUMN=64 can prevent reading an otherwise valid wide schema; a synthetic regression test confirms this is a configured limit, not proof of database corruption. CURRENT_TIMESTAMP remains outside the function allowlist. Query deadlines, row/byte limits and all read-only enforcement remain unchanged.

## Difficulty breakdown

| Difficulty | Cases | Full schema | Linking | Corrected |
|---|---:|---:|---:|---:|
| unavailable | 100 | 0.280 | 0.240 | 0.260 |

## Selection and reproduction

Full schema is the headline setting selected from development outcomes before inspecting final answers or metrics. Its paired development differences were uncertain. The interactive application's existing linking/correction workflow remains experimental, not a claimed accuracy improvement.

Generation uses local llama3.1:8b with 8,192 context tokens and at most 400 generated tokens per attempt. The three settings use the same generation parameters and a 10-second worker execution deadline. Full schema/linking use one attempt; corrected uses at most three.

[Portable comparison identities and intervals](v2-comparison-development.json) · [Failure diagnostics](failure-analysis-development.json) · [Development decision](development-decision.json) · [Commands](REPRODUCE.md).
