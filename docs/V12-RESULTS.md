# V12 results: stopped at the development gate

## Decision

**No V12 arm qualified; V12 is not promoted.** The frozen development decision (`artifacts/v12/sql-agent/development-decision.json`) records no winner. The 1,033-case locked BIRD dev holdout therefore remains unopened, and Qwen V5 remains the default.

## Development results (200 V6 development cases, 11 BIRD dev databases)

All arms ran on Modal A10G with identical runtime settings. The model digest `dae161e27b0e…` is the same one V5 used.

| Arm | EX | Lift vs control | Paired 95% CI (11 database groups) | Completion | p95 | Fixed / broke vs control |
|---|---:|---:|---|---:|---:|---:|
| Control (`qwen_v5`) | 0.375 | — | — | 0.955 | 3.88 s | — |
| Rules only | **0.395** | +0.020 | [−0.050, +0.084] | 0.920 | 3.92 s | 15 / 11 |
| Notes only | 0.360 | −0.015 | [−0.070, +0.035] | 0.915 | 4.70 s | 10 / 13 |
| Full | 0.390 | +0.015 | [−0.035, +0.065] | 0.920 | 4.54 s | 18 / 15 |

The gate required an EX lift of at least +0.03 with completion no more than 0.02 below control. Every V12 arm missed both. Latency (at most 2× control) and database integrity (zero byte-hash changes in every arm) passed.

**Reproduction check:** the control scored 0.375 on the same 200 cases where V6 recorded 0.370 for Qwen V5, with a different runtime.

## What happened

- **Rules helped slightly but cost completion.** The conventions fixed 15 control failures, typically text-number cleaning and `DISTINCT` listings, and broke 11 previously correct answers. They also introduced 10 new application failures, mostly `schema_mismatch` errors where the model referenced a column on the wrong alias.
- **Notes introduced new schema errors.** Notes list columns as raw, table-qualified names such as `district.A3` and `frpm.Enrollment (K-12)`. The model copied these into queries that alias the table or leave identifiers with spaces unquoted, producing `no such column` and parse errors. In addition, 110 of 200 prompts hit the 6,000-byte notes cap, so notes were truncated in more than half the cases.
- **Effect differed by difficulty.** With notes, simple questions improved (0.48 → 0.53), but moderate questions fell (0.33 → 0.26); moderate questions are where multi-table joins, and therefore alias conflicts, occur.

## Interpretation

The audit finding stands: grounding in stored values is a real failure source. This delivery of that grounding is flawed, though. Unquoted, table-qualified identifiers and truncated notes create new schema errors that cancel the lookups they fix. At n = 200, the ±0.02 movements are within noise. V12 is recorded as a negative result caused by the implementation, not as evidence that value grounding cannot help.

## If continued (not run)

A follow-up would change only the note format, with a new frozen protocol:
1. Print identifiers exactly as they must appear in SQL (quoted like the DDL), without table prefixes that conflict with aliases.
2. Include only columns whose names, descriptions, or values overlap the question and evidence, with a tighter budget so nothing is truncated.
3. Keep the five conventions.

Because the development set has now been used for V12, any follow-up selection on it carries additional selection risk. The locked final remains unopened and would still provide an unbiased test.

## Cost and integrity

Charged GPU time for smoke and development was about 2,630 GPU-seconds across four A10G containers, estimated at **$0.80** in `artifacts/v12/compute/modal-ledger.json`. Modal billing is authoritative, and setup CPU time is not included. No API model was used, reference SQL never entered prompts, and no database changed.
