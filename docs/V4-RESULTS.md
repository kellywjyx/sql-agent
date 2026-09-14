# V4 SQL compatibility and correctness results

V4 retains the read-only `sqlite-readonly-v3` policy: 256 schema columns, 64 returned columns, SQLGlot validation, SQLite read-only connections and authorizer, extension denial, deadlines, and row/byte limits. Valid empty results remain successful executions.

The 25-case development pilot covered at least five databases:

| Local model | Execution accuracy | Reference coverage | Completion | p95 |
|---|---:|---:|---:|---:|
| llama3.1:8b | 0.320 | 0.960 | 0.840 | 10.84 s |
| qwen2.5-coder:7b-instruct | 0.600 | 0.960 | 0.920 | 8.72 s |

The Qwen-Coder pilot was promising, but it did not enter development or final evaluation. The frozen selection script compared database bytes with an upstream dataset-record hash and interpreted the expected difference as a safety failure. This caused a conservative false stop. Selection was not rewritten after final exposure. A separate post-campaign audit verified unchanged bytes for all 55 databases.

The Llama control ran once on the fresh 200-case internal BIRD training-database holdout:

| Execution accuracy | Reference coverage | Completion | First attempt | Correct recovery | Timeout | p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.230 | 0.965 | 0.800 | 0.240 | 0.050 | 0.060 | 20.77 s |

Forty application failures remain in the denominator. This is an internal holdout result, not an official BIRD benchmark or leaderboard submission. Successful execution is reported separately from correctness. Qwen-Coder's pilot is diagnostic evidence and is not presented as a final score.

Generation now receives a deterministic report for requested projection, aggregation, grouping, filters, joins, and ordering. Parser, safety, execution, timeout, and coverage diagnostics can guide correction within three total attempts. No benchmark-only security exemption was added. API cost was `$0`.
