# SQL Agent V8 results

## Decision

**V8 stops after exposed development and is not promoted. Qwen V5 remains the
public default.** Quantization is closed: V8 uses Arctic Q4 only. No locked-final,
Talon, Q5, external API, or multi-candidate run was performed.

## Oracle-help decomposition

The same 40 exposed V7 reproduction cases were rescored under the immutable
no-evidence and oracle-evidence captures.

| Group | Cases | Meaning |
|---|---:|---|
| A | 9 | No evidence wrong, oracle correct: evidence-limited |
| B | 12 | Both correct: evidence unnecessary |
| C | 19 | Both wrong: reasoning-limited |
| D | 0 | No evidence correct, oracle wrong |

All cases received deterministic evidence-type and recoverability labels. These
labels and gold-derived structures are scoring-only and never enter prompts.

## CPU evidence gate

On the nine Group-A cases, the combined typed packet achieved:

| Metric | Result | Gate |
|---|---:|---:|
| Literal recall | 0.857 | >=0.600 |
| Value-to-column accuracy | 0.923 | >=0.800 |
| Evidence precision | 0.923 | >=0.700 |
| Table recall | 1.000 | diagnostic |
| Column recall | 0.682 | diagnostic |
| Transformation recall | 0.000 | diagnostic failure |

The cache/router raised literal recall well above V7's 0.283 headline while
remaining precise on the intended evidence-limited subset. Transformation
reconstruction remains unimplemented in a useful form and is reported as zero.

## Matched 20-case evidence ablation

The pilot contains all nine Group-A cases plus six B and five C controls. Every
generated condition uses the same Arctic Q4 model, reference-like prompt, schema,
decoding, evaluator, and read-only policy.

| Condition | EX | Completion | p95 | Recoveries / regressions | Oracle-gap recovery |
|---|---:|---:|---:|---:|---:|
| No evidence control | 0.300 | 0.900* | 12.92 s* | — | 0% |
| Generated literals | 0.400 | 0.950 | 12.97 s | 3 / 1 | 22.2% |
| Generated descriptions | 0.200 | 0.950 | 12.52 s | 0 / 2 | -22.2% |
| Generated normalization | 0.350 | 0.950 | 12.77 s | 2 / 1 | 11.1% |
| All generated evidence | **0.450** | 0.950 | 14.01 s | **5 / 2** | **33.3%** |
| Oracle evidence ceiling | 0.750 | 0.975* | 13.00 s* | — | 100% |

`*` Control completion and latency are the immutable 40-case run headlines;
EX is paired on the exact 20 pilot IDs. Generated-all improved by 0.150, but its
database-grouped 95% interval was [-0.120, 0.444]. This is a passing development
gate and a high-uncertainty pilot, not a quality claim.

Description-only evidence is a negative result. Broad source-backed descriptions
can distract the local specialist when not paired with precise values.

## Broader exposed-development result

The frozen generated-all route then ran once on all 40 exposed cases.

| Metric | No evidence | Generated | Oracle |
|---|---:|---:|---:|
| Execution accuracy | 0.300 | **0.350** | 0.525 |
| Oracle Evidence Gap Recovery | 0% | **22.2%** | 100% |

Generated evidence produced five wrong-to-correct recoveries and three
correct-to-wrong regressions. Its +0.050 paired delta has a database-grouped 95%
interval of [-0.100, 0.216], below the frozen +0.075 continuation target.
Completion was 0.950, p95 latency 13.74 seconds, and every database hash was
unchanged.

The group breakdown explains the weak aggregate:

| Group | No evidence EX | Generated EX | Oracle EX |
|---|---:|---:|---:|
| A: evidence-limited (9) | 0.000 | **0.444** | 1.000 |
| B: evidence unnecessary (12) | 1.000 | **0.750** | 1.000 |
| C: reasoning-limited (19) | 0.000 | **0.053** | 0.000 |

The system reconstructed useful facts for four of nine evidence-limited cases,
but unnecessary packets broke three already-correct cases. Across all 40 cases,
literal recall was 0.481 and evidence precision 0.241; the strong Group-A router
metrics do not generalize to indiscriminate evidence injection.

A release smoke test then found one deterministic contributor to packet noise:
the evaluated cache v1 converted native SQLite INTEGER/REAL values to strings
before pattern classification and consequently labeled ordinary numeric columns
as `numeric_text_cast`. Cache v2 fixes the type check and invalidates old cache
identities. The fix is covered by offline tests but was **not** rerun on these
development outcomes, so no model-quality gain is claimed for it.

## Engineering conclusion

V8 establishes a narrower causal result: database-side evidence reconstruction
can recover part of the human-evidence benefit, but evidence activation and
transformation reasoning remain unsolved. A future version should predeclare and
calibrate an evidence-activation rule on new exposed development material before
running more SQL generation. It should not tune that rule against these 40
outcomes.

Five model jobs used 1,179 charged seconds under the one-hour ledger. API charges
were $0. Seventy-five offline/security tests pass. The independently imported
0.8.0 wheel has SHA-256
`c7a3b361328f55ffa12f4e16a99f84945758a61ac541de844eabbee110aa46b2`.
The 1,033-case locked final remains unopened.
