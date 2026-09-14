# V13 results: stopped at the development gate

## Decision

**No V13 arm qualified; V13 is not promoted.** The frozen decision (`artifacts/v13/sql-agent/development-decision.json`) records no winner. The notes-plus-conventions arm cleared the accuracy and latency conditions but missed the completion condition by one percentage point. The 1,033-case locked BIRD dev holdout remains unopened, and Qwen V5 remains the default.

## Development results (same 200 V6 development cases)

The control is the reused V12 development control run (`20260914T073433.908471Z-7252659f`), with the same cases, code path, model digest `dae161e27b0e…`, runtime, and deterministic decoding.

| Arm | EX | Lift vs control | Paired 95% CI (11 database groups) | Completion | p95 | Fixed / broke |
|---|---:|---:|---|---:|---:|---:|
| Control (`qwen_v5`) | 0.375 | — | — | 0.955 | 3.88 s | — |
| V13 notes (`qwen_v13_notes`) | 0.380 | +0.005 | [−0.035, +0.045] | 0.950 | 4.11 s | 10 / 9 |
| V13 full (`qwen_v13`) | **0.420** | **+0.045** | [−0.020, +0.101] | 0.925 | 4.39 s | 22 / 13 |

**Frozen gate:**
- **Full arm:** passed EX lift (+0.045 ≥ +0.03), latency (4.39 s ≤ 2× 3.88 s), and database integrity (zero changes). It failed completion: 0.925 is 0.030 below control, and at most 0.020 was allowed.
- **Notes arm:** passed completion but not EX lift.

## Comparison with V12 (same cases)

| Arm | V12 EX / completion | V13 EX / completion |
|---|---|---|
| Notes | 0.360 / 0.915 | 0.380 / 0.950 |
| Notes + conventions | 0.390 / 0.920 | 0.420 / 0.925 |

Quoted, unqualified identifiers removed most note-induced failures in the notes-only arm: completion went from 0.915 to 0.950. They also lifted the full arm's accuracy by 3 points.

## What still fails

- **Remaining schema errors:** the full arm has 11 application failures beyond the control's, mostly `schema_mismatch`, where a column is referenced on the wrong alias after correction. The conventions appear to encourage more multi-table rewrites, and those still break.
- **Truncation:** 97 of 200 notes still dropped at least one relevant entry at the 3,000-byte budget, so the relevance filter is broader than intended.
- **Noise:** the +0.045 interval includes zero. With n = 200 this is suggestive, not established.

## Interpretation

V13 is the best development result since V5: +4.5 points and the most fixed cases (22). It confirms that the V12 failure was about formatting. It is still not a promotable result. It missed a completion gate that was frozen in advance, the accuracy gain is within noise, and V13 was designed after seeing V12's failures on the same development cases, which makes its development numbers optimistic. The gate was not relaxed, and the locked final was not opened.

## Cost and integrity

The combined V12 and V13 GPU estimate is **$1.22** (`artifacts/v12/compute/modal-ledger.json`); Modal billing is authoritative. No database changed in any arm, gold self-consistency was 1.0, no API model was used, and reference SQL never entered prompts.
