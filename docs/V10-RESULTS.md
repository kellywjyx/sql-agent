# V10 Results: Error-Conditioned Semantic Correction

## Decision

V10 is implemented and **not promoted**. The fresh-pilot gate failed, so the
selective Qwen corrector, broader exposed campaign, and 1,033-case locked final
were not run. Qwen V5 remains the public default.

## Error memory and calibration

The frozen memory contains 303 schema-independent records:

- 293 execution-changing synthetic mutations from 150 BIRD-training cases;
- 10 single-family natural failures from a 30-case Arctic scoped-evidence capture;
- 108 synthetic operation, 4 grain, 88 predicate, and 93 join records.

The natural capture scored 0.300 EX, 0.933 completion, and 18.92-second p95.
It is diagnostic training material, not benchmark evidence.

Calibration used 70 inspected/development cases. Only predicate correction met
the family gate:

| Family | Flags | Precision | Wrong attempts | Repair success | Correct-query regressions | Enabled |
|---|---:|---:|---:|---:|---:|---|
| Operation | 1 | 1.000 | 0 | n/a | 1 | No |
| Grain | 0 | n/a | 0 | n/a | 0 | No |
| Predicate | 2 | 1.000 | 2 | 1.000 | 0 | **Yes** |
| Join | 1 | 0.000 | 0 | n/a | 0 | No |

This calibration deliberately favors abstention. Operation was disabled despite
a correct label because its only intervention would have modified a correct query.

## Fresh 40-case pilot

| Configuration | EX | Completion | p95 | Corrections attempted |
|---|---:|---:|---:|---:|
| Frozen Arctic Q4 + scoped evidence | 0.300 | 1.000 | 14.58 s | n/a |
| + calibrated deterministic correction | 0.300 | 1.000 | 15.17 s estimated end-to-end | 0 |
| Same-set Qwen V5 control | **0.375** | 0.950 | **5.28 s warm** | n/a |

The deterministic replay had 12 correct-to-correct and 28 wrong-to-wrong cases,
with zero recoveries and zero regressions. Its paired database-grouped EX delta
against Arctic was 0.000 with a 95% interval of [0.000, 0.000]. Qwen's +0.075
delta against Arctic had a 95% interval of [-0.048, 0.205], so the small pilot
does not establish a statistically reliable improvement.

## Gate outcome

V10 failed because:

- EX lift was below +0.075;
- net corrections were below three;
- fresh-pilot detector precision was unscorable because nothing was flagged;
- fresh-pilot repair success was unscorable because nothing was attempted.

Completion, latency, security, and database-integrity requirements passed. The
correct response to the empty intervention set is not to lower the threshold or
invoke a generic second model call. The selective Qwen corrector therefore stayed
off.

## Interpretation

The correction machinery is safe and localized, but the calibrated predicate
detector transferred too narrowly to the fresh pilot. The memory size alone did
not create useful interventions, and the other three families lacked safe
calibration support. This rejects the practical V10 hypothesis in its current
form: retrieval-conditioned deterministic correction does not improve the frozen
Arctic scoped branch on fresh cases.

The same-set Qwen result also reinforces the existing default decision. Although
the 40-case estimate is noisy, Qwen was more accurate and substantially faster
than Arctic on this slice. Adding further inference layers to Arctic is not
justified by these results.

## Resource and safety record

- Charged local model time: 1,010 seconds of the 3,600-second V10 ceiling.
- External API charges: $0.
- Database hash mismatches: 0 across all three fresh-pilot conditions.
- Locked final opened: no.
- Gold SQL in runtime correction memory or prompts: no.

The next technically justified branch, if continued, is targeted adaptation or a
larger high-quality detector-learning set. It should be proposed separately and
must first show that detector recall can increase without sacrificing intervention
precision. More probes, generic candidates, and unbounded review remain unsupported.
