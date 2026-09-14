# V6 measured results

## Status

**Implementation, offline verification, model pilots, and development selection are complete. The locked final evaluation was deliberately not opened because measured runtime made the two required final passes exceed the frozen four-hour budget.** This is a budget-gate outcome, not a failed or completed final evaluation, and no V6 model is promoted.

The final split contains 1,033 unique untouched BIRD-development question groups. The plan's provisional 1,037 count was corrected from authoritative upstream IDs: excluding exactly 500 prior Mini-Dev records leaves 1,034 records, with one duplicate question group retained only at its lower upstream index. The final dataset hash is `08e4ff1c288d11b987b3673a1130ce5af1d3e508294359bcaeb0af4294a9b41a`.

## Pilots: 30 exposed development cases

| Configuration | EX | Completion | p95 latency | Outcome |
|---|---:|---:|---:|---|
| Qwen V5 frozen control | 0.333 | 0.933 | 5.80 s | Control |
| Qwen M-Schema, profiled values | 0.200 | 0.833 | 4.82 s | Negative prompt/schema ablation |
| Arctic direct | **0.467** | **0.967** | 15.10 s | Qualified |
| Arctic plan-first | **0.467** | **0.967** | 16.83 s | Same accuracy, slower |
| XiYan direct | 0.367 | 0.867 | 4.66 s | Failed 0.95 completion gate |

The first Qwen M-Schema capture exposed null target columns in implicit SQLite foreign keys and false single-character value matches. It remains preserved as diagnostic evidence. After deterministic fixes, a new immutable 30-case capture completed without those crashes and is the result reported above.

Arctic direct improved pilot EX by 13.3 points over Qwen and passed the 0.95 completion gate. Its 0.467 EX remained below 0.65, which activated the predeclared XiYan fallback. XiYan was faster but did not qualify. Plan-first did not improve Arctic, so direct advanced.

## Development selection: 200 inspected Mini-Dev cases

| Configuration | EX | Capture coverage | Reference coverage | p95 latency | Database changes |
|---|---:|---:|---:|---:|---:|
| Qwen V5 control | 0.370 | 0.940 | 0.960 | 6.07 s | 0 |
| Arctic direct | **0.465** | **0.950** | 0.960 | 15.50 s | 0 |

Arctic improved development EX by 9.5 points and met the 20-second latency limit. It became the frozen final candidate. Adaptive two-candidate generation was not enabled: the completed single-candidate captures provide no five-point candidate-oracle gain, so the predeclared adaptive gate was not satisfied.

An audit found that the initial frozen V5 control capture had generated a third correction candidate in two development cases. Those artifacts and `selection.json` remain preserved but are superseded. A corrected control capped at two candidates produced the figures above, and `selection-v2.json` freezes Arctic using the compatible policy.

These are selection results on inspected BIRD development material. They are neither final claims nor an official benchmark score. SQL-specialized model training exposure may overlap BIRD, and development/final share schemas even though question and near-duplicate groups are isolated.

## Compute-budget stop

Completed model jobs charged 4,957 of 14,400 seconds, leaving 9,443 seconds. This includes the policy-corrected Qwen pilot and development reruns. Same-set measured wall rates project:

| Required final run | Cases | Projected time |
|---|---:|---:|
| Qwen V5 control | 1,033 | 3,564 s |
| Arctic direct winner | 1,033 | 12,086 s |
| **Total** | | **15,650 s** |

The required sequence is projected to exceed the remaining ledger by 6,207 seconds (1.72 hours), before contingency. The protocol says not to start a run projected to consume reserved final capacity and not to strand an incomplete locked capture. Therefore `final_capture_started` is false in `artifacts/v6/sql-agent/budget-stop.json`.

A separate authorization for at least 6,207 additional model-job seconds plus contingency is required to run the two locked-final passes. Until then, Qwen V5 remains the application default and the existing V5 result remains the public portfolio evidence.

## Reliability and security evidence

- 52 offline, API, planning, profiling, semantic, and security tests pass.
- Implicit foreign-key targets resolve to unambiguous referenced primary keys instead of crashing.
- Single-character categorical profile values require explicit literal evidence and no longer match arbitrary prose.
- Every pilot and development database has identical pre/post SHA-256 bytes.
- Gold SQL is confined to scoring callbacks; prompts and correction diagnostics do not contain it.
- Arctic and XiYan run locally through Ollama; API cost is `$0`.
- Exact model revisions, GGUF hashes, sizes, licenses, and Ollama digests are recorded in [V6-MODEL-ASSETS.json](V6-MODEL-ASSETS.json).

## Interpretation

The SQL-specialized Arctic checkpoint improved semantic construction on development data, while M-Schema alone and XiYan did not. The strongest defensible V6 claim today is a development-selected local improvement under unchanged security controls. It is not yet evidence for the target `≥0.65` final EX, `≥0.97` completion, or promotion gate because the locked final evaluation has not run.
