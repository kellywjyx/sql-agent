# V5 results: local text-to-SQL under a fixed budget

## Outcome

V5 completed within the frozen local-job budget and promoted **Qwen2.5-Coder 7B with the full bounded schema** for evaluated correctness. The final result is an internal BIRD-training holdout experiment, not Mini-Dev, an official BIRD submission, or a leaderboard score.

| Frozen final metric (200 cases, 11 unseen databases) | Llama full-schema control | Qwen full-schema candidate |
|---|---:|---:|
| Execution accuracy | 0.335 | **0.545** |
| Execution completion | 0.815 | **0.935** |
| First-attempt accuracy | 0.305 | **0.520** |
| Correct recovery fraction | 0.030 | 0.025 |
| Timeout fraction | 0.035 | **0.025** |
| Reference coverage | 1.000 | 1.000 |
| p95 end-to-end latency | 12.50 s | **8.35 s** |

The paired execution-accuracy difference is **+0.210**. A 2,000-resample paired bootstrap grouped by database gives a 95% interval of **[+0.085, +0.324]**. The candidate passed every frozen promotion rule: EX at least 0.30, at least five points over control, positive grouped interval, completion at least 0.80, reference coverage at least 0.95, p95 below twice control, and unchanged SQLite bytes.

The final failure counts remain visible: Llama produced 67 correct, 96 wrong-result, and 37 application-failure cases; Qwen produced 109 correct, 78 wrong-result, and 13 application-failure cases. Successful execution is never scored as correctness by itself.

## Selection evidence

The answer-free retrieval gate ran before model selection on all 100 V3 development cases:

| Retrieval metric | Result | Required |
|---|---:|---:|
| Required-table recall | 1.000 | 0.980 |
| Valid required-column recall | 0.987 | 0.950 |
| Foreign-key-path coverage | 1.000 | reported |

Nine gold-column references were absent from their supplied SQLite schemas and were reported separately rather than counted as retriever misses. Gold SQL was parsed only by scoring code after application capture.

The corrected 25-case pilot showed:

| Candidate | EX | Completion | Interpretation |
|---|---:|---:|---|
| Llama full schema | 0.320 | 0.800 | Control |
| Qwen full schema | **0.640** | **0.920** | Pilot winner |
| Qwen hybrid retrieval | 0.600 | 0.880 | Strong, but pilot-dominated by full schema |
| Qwen hybrid plus deterministic review | 0.240 | 0.800 | Review overcorrected valid candidates |

The 100-case development selection then measured Qwen full schema at **0.430 EX / 0.850 completion**, versus Llama at **0.230 / 0.780**. The candidate was frozen before the new final holdout was opened.

## Negative findings

Semantic retrieval met its recall targets but did not improve generation on the pilot. Most schemas already fit the 10 KB prompt budget, so removing schema context offered little benefit and occasionally changed Qwen’s chosen joins or projections. Hybrid retrieval remains available for large or unfamiliar schemas and is the API-compatible default requested for the interactive demonstration, but it is not presented as the winning evaluated configuration.

The deterministic review layer also reduced pilot correctness. Early coverage rules treated explanatory benchmark evidence as query intent and forced unnecessary aggregation or filters. V5.1 corrected those false requirements, stopped duplicate SQL without re-execution, and allowed a third call only for a changed failure. Automatic semantic correction remains disabled in the default interactive request path; coverage diagnostics are still returned for inspection.

## Integrity, compute, and identity

- All final databases had matching pre/post SHA-256 byte hashes.
- Read-only policy `sqlite-readonly-v3` remained unchanged: 256 schema columns, 64 returned columns, authorizer, validation, deadlines, extension/attachment denial, and row/byte limits.
- Final dataset identity: `2cbaacef387bb2878524feecd8a58e3b04ef9892edc5693d43e93c73b233f95f`.
- Llama digest: `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`.
- Qwen digest: `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364`.
- BGE revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- The cumulative ledger charged **5,555 seconds (1.54 hours)**, including preserved superseded V5.0 runs, below the 6,300-second ceiling.
- API expenditure: **$0**.

The first orchestrator attempt wrote an immutable budget-stop record when its conservative reserve rejected another development job. No limit was extended. V5.1 reused the same ledger, pruned pilot-dominated candidates, preserved final capacity, and completed both final captures.

## Artifacts

The internal immutable campaign report is `artifacts/v5/sql-agent/final-report.json`; selection is `artifacts/v5/sql-agent/selection.json`; the cumulative compute record is `artifacts/v5/compute/ledger.json`. These internal artifacts contain machine paths and withheld benchmark material and must go through an allowlisted public export before website or GitHub publication.
