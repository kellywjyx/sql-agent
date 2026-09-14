# SQL Agent V9 results

## Decision

**V9 stopped after the 20-case exposed pilot and is not promoted.** Qwen V5 at
0.545 on the separate V5 internal holdout remains the strongest defensible public
default. The 40-case continuation, verified semantic IR, and 1,033-case locked
final were not run.

## Transition audit

The scoring-only audit reproduced V8's five recoveries and three regressions.
Among the eight changed cases, 16 of 21 reference-relevant evidence uses were
mechanically correct (utilization accuracy 0.762). Seventy facts were ignored,
six irrelevant facts affected SQL, and four value uses contradicted reference
intent. Changed components included predicates (4), joins (3), projection (1),
bindings/expressions (3), and syntax/extraction (1). This supports the hypothesis
that evidence correctness and evidence use are separate problems.

## Matched pilot

All conditions use the same 20 exposed cases, Arctic Q4 model/digest,
reference-like prompt, schema, decoding, evaluator, and read-only policy.

| Condition | EX | Completion | Recoveries / regressions | Utilization accuracy | p95 |
|---|---:|---:|---:|---:|---:|
| Immutable no-evidence control | 0.300 | — | — | — | — |
| Cache-v2, unscoped | 0.350 | 0.950 | 5 / 4 | 0.588 | 14.18 s |
| Scoped evidence | **0.450** | **1.000** | **4 / 1** | **0.647** | **12.31 s** |
| Scoped + deterministic probes | **0.450** | **1.000** | **4 / 1** | 0.627 | 13.15 s |

Scoping improved EX by 0.10 over the matched cache-v2 control and reduced
regressions from four to one. Against no evidence its +0.15 database-grouped
95% interval was [-0.08, 0.412], so uncertainty remains high.

The probe condition executed 77 bounded observations: 48 join-cardinality, 21
value-existence, and 8 numeric-storage checks. It produced no additional correct
case, increased p95 by 0.84 seconds, and slightly worsened utilization. Verified
observations alone do not fix reasoning when the generator still chooses the
wrong operation or grain.

No condition reached the frozen pilot requirements of six recoveries and five
net recoveries. The campaign therefore stopped before verified IR and broader
evaluation. This is a failed promotion gate, not a failed or incomplete run.

## Reliability and cost

All three captures completed. The compute ledger charged 634 model-job seconds
against 3,600 available. Every database byte hash was unchanged, official gold
self-consistency was 1.0, and API charges were $0. Eighty-five offline/security
tests pass in the final source tree.

## Conclusion

V9 provides a useful positive and negative result. Explicit role scoping reduces
evidence-induced regressions and restores the V8 pilot score after the cache-v2
cleanup. Small verification probes, however, do not improve execution accuracy.
The unresolved bottleneck is converting grounded facts into the correct result
grain, operations, and query structure. Further work should use the recorded
evidence-conditioned reasoning errors as contrastive development examples rather
than add more probes, candidates, or generic prompt context.
