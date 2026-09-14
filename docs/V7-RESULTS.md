# SQL Agent V7 results

## Decision

**V7 is not promoted. Qwen V5 remains the public default.** The planned stop gate
worked: the three-path pilot did not reach 0.65 candidate-oracle EX, so no full
60-case calibration, 100-case selection, 200-case regression replay, Talon run,
or locked-final evaluation was started.

## Arctic reproduction audit

All conditions use the existing local Arctic-Text2SQL-R1-7B Q4_K_M asset on the
same 40 exposed, database-stratified cases. BIRD evidence is oracle evaluation
context; generated evidence is the only public-compatible route.

| Condition | EX | Completion | p95 | Literal recall |
|---|---:|---:|---:|---:|
| Frozen V6 prompt + oracle evidence | 0.525 | 0.900 | 15.38 s | 0.000* |
| Reference-like prompt + oracle evidence | 0.525 | 0.975 | 13.00 s | 0.271 |
| Reference-like prompt, no evidence | 0.300 | 0.900 | 12.92 s | 0.000 |
| Reference-like prompt + generated evidence | 0.300 | 0.950 | 12.89 s | 0.259 |

`*` The frozen V6 path does not serialize evidence into the V7 packet, so packet
literal recall is not a meaningful measure for that row.

The reference-like prompt improved completion and latency but did not improve EX.
Oracle evidence added 22.5 points relative to no/generated evidence on this small
split. The local reference-like Q4 result is 16.4 points below the model card's
reported 0.689 BF16 BIRD-dev result; the datasets and inference stacks are not
directly comparable. This activated a separately approved 20-case Q5 matched
diagnostic, reported below.

The checksum-pinned Mini-Dev `calculate_ex` function was executed in isolation.
Gold-versus-gold self-consistency was 1.000 in all four runs. Local-versus-official
prediction agreement ranged from 0.943 to 1.000 because the local comparator can
respect ordered-result metadata while the upstream set comparator ignores order
and duplicates. This disagreement is reported, not hidden.

## Three-path candidate pilot

| Metric | Result | Gate |
|---|---:|---:|
| Selected EX | 0.350 | diagnostic |
| First-candidate EX | 0.350 | diagnostic |
| Candidate-oracle EX | 0.500 | ≥0.650 to continue |
| Completion | 1.000 | ≥0.970 |
| Table recall | 1.000 | ≥0.980 |
| Column recall | 0.980 | ≥0.950 |
| Literal recall | 0.283 | ≥0.850 |
| p95 latency | 49.03 s | ≤20 s interactive |
| Database changes | 0 | 0 |
| API charges | $0 | $0 |

The 15-point gap between selected and oracle EX demonstrates complementary
candidates, but a perfect selector could still reach only 0.500 on this pilot.
Every selected answer required review, and three-path latency was unsuitable for
the public interface. The dominant measured bottlenecks are generated evidence
for predicate literals and generator semantics, not schema-table coverage.

## Matched Q5 quantization diagnostic

The diagnostic reused the first 20 reproduction cases, reference-like prompt,
oracle evidence, schema serialization, decoding, SQL extraction, official scorer,
and read-only execution policy. Only `Q4_K_M` changed to `Q5_K_M`.

| Measure | Q4 | Q5 |
|---|---:|---:|
| Execution accuracy | 0.550 | 0.500 |
| Completion | 0.950 | 0.950 |
| p95 latency | 12.68 s | 13.26 s |

Paired transitions were 10 both-correct, 0 Q4-wrong/Q5-correct recoveries, 1
Q4-correct/Q5-wrong regression, and 9 both-wrong. The paired database-grouped
delta was -0.050 with a 95% interval of [-0.158, 0.000]. Q5 therefore fails the
predeclared retain rule. The remaining upstream-performance gap cannot reasonably
be attributed primarily to Q4 quantization, and no Q6, Q8, BF16, or Talon run is
justified by this result.

The single regression was `bird-mini-dev-1`: Q5 added an unrequested aggregate
projection to an otherwise equivalent minimum-consumption query, changing the
result shape. The case-level transition artifact records generated SQL and
correctness without copying gold SQL.

The next technical direction is evidence grounding: decompose oracle-help cases,
measure literals by type, value-to-column mapping, aliases, transformations, and
FK paths, then show that improved generated evidence raises single-candidate EX
before restoring multiple candidates or selector work.

## Integrity and scope

- 68 offline/security tests pass; package compilation and dependency checks pass.
- Every campaign database retained the same pre/post byte hash.
- All 180 reproduction predictions and 20 candidate-pilot predictions are
  immutable, per-case checkpointed artifacts under `artifacts/v7/sql-agent/`.
- Model/API spend was $0. Charged local model-job time was 2,719 seconds across
  smoke, reproduction, and candidate-pilot jobs.
- The 1,033-case locked final remains unopened.

These are local portfolio experiments on exposed development material, not an
official BIRD submission or a claim that the Q4 model reproduces upstream BF16
results.
