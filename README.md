# sql-agent

Question → schema/value linking → SQL generation → AST guard → read-only worker → bounded correction

## Status

**V13 SQL-ready column notes stopped at the frozen development gate and are not promoted.** V13 changed only the V12 note format: quoted, unqualified identifiers, question-relevant columns, and matching stored values. On the same 200 BIRD dev cases, notes plus conventions reached **0.420 EX** versus 0.375 for the reused control (+0.045, 95% CI [−0.020, +0.101]), the best development result since V5. Completion fell to 0.925 versus 0.955, missing the frozen −0.02 limit by one point. Notes alone scored 0.380 with 0.950 completion. The locked final remains unopened; Qwen V5 remains the default. See [V13 results](docs/V13-RESULTS.md) and [V13 design](docs/V13.md).

**V12 value-grounded prompting stopped at its frozen development gate and is not promoted.** Following a hand audit of all 91 V5 failures ([audit](docs/V5-FAILURE-AUDIT.md); audited EX 0.615 vs strict 0.545), V12 added BIRD column descriptions, stored-value examples, storage-format tags, and five SQL conventions to the frozen V5 path. On 200 BIRD dev cases (Modal A10G, same model digest), the control scored 0.375, rules 0.395, notes 0.360, and full 0.390. Every arm lost 0.035–0.040 completion to new schema-mismatch errors (unquoted, table-qualified identifiers in notes; 110/200 notes truncated), and every paired interval included zero. The 1,033-case locked final remains unopened; Qwen V5 remains the default. See [V12 results](docs/V12-RESULTS.md) and [V12 design](docs/V12.md).

**V11.1/V11.2 direct QLoRA adaptation ran on rented Modal GPUs and stopped at its frozen pilot gate; it is not promoted.** On an A100 the 7B recipe peaked at 12.1 GB, confirming the local 8 GB limit. A token-count bug in the frozen V11 data prep (it measured a BatchEncoding's keys, so 10% of training and 28% of validation examples were silently over 2,048 tokens) was found through NaN validation loss, fixed with tests, and retrained as V11.2 with every quality gate unchanged. On 60 development cases, the adapters scored 0.283 (recipe A, 95% CI of the lift [-0.225, -0.015]) and 0.367 (recipe B, [-0.068, +0.038]) versus 0.383 for the base model, so full training was not run. Database hashes were unchanged and Modal spend was about $1.63. Qwen V5 remains the public default. See [V11.1](docs/V11.1.md) and [V11.2](docs/V11.2.md).

**V11 direct QLoRA adaptation is implemented and stopped at its mandatory hardware preflight.** The pinned Qwen2.5-Coder-7B-Instruct snapshot (revision `dcc04bb77301e6368aa24a609c6ea496b70503e8`, 15.24 GB) passed checksum verification. The database-disjoint BIRD roles were frozen at 1,200 training, 150 validation, and 200 development cases, with 100% of selected inputs fitting the 2,048-token limit and no human BIRD evidence in prompts. The exact CUDA stack was installed, but the RTX 4060 Ti exposed 7,406,092,288 free bytes, 110,100,480 bytes below the strict 7 GiB gate. No model was loaded, no training or evaluation began, GPU training time was zero, and the locked final remains unopened. Per protocol there is no offload, smaller-model fallback, or relaxed threshold. Qwen V5 remains the public default. See [V11 results](docs/V11-RESULTS.md) and [V11 design](docs/V11.md).

**V10 selective semantic correction is implemented and stopped at its frozen fresh-pilot gate.** A 303-record, schema-independent error memory combines 293 execution-changing synthetic mutations with 10 natural Arctic failures. Calibration enabled only the predicate family: it achieved 1.000 precision and 1.000 repair success on two inspected wrong queries, while operation, grain, and join correction abstain. On the untouched 40-case V10 pilot, frozen Arctic scoped evidence scored 0.300 EX and the deterministic correction replay also scored 0.300 because the calibrated detector correctly made zero interventions. Same-set Qwen V5 scored 0.375. The correction gate failed, selective Qwen correction was not run, the 1,033-case locked final remains unopened, database hashes were unchanged, and API spend was $0. Qwen V5 remains the public default. See [V10 results](docs/V10-RESULTS.md) and [V10 design](docs/V10.md).

**V9 execution-grounded semantic reasoning stopped at its frozen 20-case pilot gate and is not promoted.** The cache-v2 preflight scored 0.350 EX with five recoveries and four regressions. Adding explicit SQL-role scope raised EX to 0.450, reduced regressions to one, reached 1.000 completion, and held p95 latency to 12.31 seconds. Seventy-seven deterministic read-only probes produced no additional EX gain and slightly reduced measured evidence-utilization accuracy. No condition reached the frozen six-recovery/five-net-recovery gate, so verified IR, the 40-case campaign, and the 1,033-case locked final were not run. Qwen V5 remains the public default, database hashes were unchanged, and API spend was $0. See [V9 results](docs/V9-RESULTS.md) and [V9 design](docs/V9.md).

**V8 database-grounded evidence reconstruction is complete on exposed development and is not promoted.** A typed offline cache raised literal recall to 0.857 with 0.923 evidence precision on the nine evidence-limited diagnostic cases. On a matched 20-case pilot, generated evidence raised Arctic Q4 EX from 0.300 to 0.450 and recovered 33.3% of the oracle-evidence gap. On all 40 exposed cases, however, EX reached only 0.350 versus 0.300 without evidence and 0.525 with oracle evidence; the +0.050 interval [-0.100, 0.216] missed the frozen +0.075 continuation gate. Evidence recovered 4/9 evidence-limited cases but regressed 3/12 evidence-unnecessary cases. V8 stopped, Qwen V5 remains the public default, API spend was $0, and the locked final remains unopened. See [V8 results](docs/V8-RESULTS.md) and [V8 design](docs/V8.md).

**V7 semantic-grounding implementation and gated pilots are complete; V7 is not promoted.** The 40-case Arctic reproduction audit found 0.525 EX with either the frozen V6 prompt or the reference-like prompt when oracle BIRD evidence was supplied, versus 0.300 without evidence and 0.300 with generated evidence. A separately approved matched 20-case Q5 diagnostic scored 0.500 versus 0.550 for Q4, with zero recoveries and one regression, so quantization work stopped. The 20-case three-path pilot reached 0.350 selected EX and 0.500 candidate-oracle EX, with 1.000 completion but 49.0-second p95 latency. It failed the predeclared 0.65 pilot oracle gate, so the 60/100-case campaigns and inspected regression replay were not run. Qwen V5 remains the public default and the 1,033-case locked final remains unopened. See [V7 results](docs/V7-RESULTS.md) and [V7 design](docs/V7.md).

**V6 implementation and development selection are complete; locked-final evaluation is budget-blocked and no V6 model is promoted.** Arctic direct reached 0.465 EX versus 0.370 for Qwen V5 on the frozen 200-case development set, with 0.950 capture coverage and 15.50-second p95 latency. Measured rates project the two required 1,033-case final runs 6,207 seconds beyond the remaining four-hour ledger, so the final set remains unopened. Qwen V5 and the existing V5 public evidence remain the default. See [V6 results](docs/V6-RESULTS.md), [V6 design](docs/V6.md), and [model provenance](docs/V6-MODEL-ASSETS.json).

**V5 completed within its fixed 1.75-hour local-model budget.** On a new internal 200-case, 11-database BIRD-training holdout, Qwen full schema reached **0.545 execution accuracy and 0.935 completion**, versus **0.335 and 0.815** for Llama. The paired database-grouped improvement was +0.210 with a 95% interval of [+0.085, +0.324]; all database bytes were unchanged and API spend was $0. Hybrid retrieval met its recall gates but scored 0.600 versus 0.640 for full schema in the 25-case pilot, so it remains an application feature and documented negative ablation rather than the promoted evaluation configuration. See [V5 results](docs/V5-RESULTS.md) and [V5 design and reproduction](docs/V5.md).

**v4 compatibility campaign completed with a recorded selection failure.** The fresh internal 200-case result is 0.230 execution accuracy with 0.800 completion under the unchanged read-only policy. Qwen-Coder scored 0.600 versus 0.320 for Llama on the 25-case pilot, but a conservative database-hash comparison bug stopped it before development and final evaluation. The pilot is diagnostic only; all 55 database byte hashes were unchanged. See [V4 results](docs/V4-RESULTS.md).

**v2 local campaign completed with failures recorded.** All three settings ran on 100 BIRD training development cases and 500 SQLite Mini-Dev final cases. Final guarded execution accuracy is 19.0% for full schema, 15.8% for linking, and 19.6% for correction; the correction advantage is inconclusive. All 55 database hashes are unchanged. [Final comparison](docs/V2-RESULTS.md) and [failure analysis](docs/FAILURE-ANALYSIS.md) explain 112 unscorable reference queries and distinguish execution recovery from correct answers. These are guarded Mini-Dev results, not leaderboard scores.

V1 completed real local smoke runs. Its measured results and limitations are preserved in [V1 results](docs/V1-RESULTS.md). Offline tests establish behavior, not model quality. Source is public at https://github.com/kellywjyx/sql-agent; hosted CI runs the offline tests on every push. Portfolio site: https://kellywjyx.github.io/portfolio-site/

## Independent Windows setup

Run from this repository directory with Python 3.13 installed. No parent workspace scripts are required.

```powershell
.\scripts\setup.ps1 -EvalsWheel C:\path\to\portfolio_llm_evals-0.4.0-py3-none-any.whl -EvalsSHA256 3f0f45d9c16bf8f7af773cd19bd0425fc2c77b61a74f86ef4f9c7ffbf2bd3b30
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pip check
```

Consumers require the versioned llm-evals wheel and checksum in `evals-dependency.json`. The wheel is published with the [llm-evals v0.4.0 release](https://github.com/kellywjyx/llm-evals/releases/tag/v0.4.0): download it and pass its path to `-EvalsWheel`. CI downloads the same file through the `LLM_EVALS_WHEEL_URL` repository variable and verifies the pinned SHA-256. It is never downloaded at application startup.

## Local assets and commands

Assets default to a local `artifacts` directory. Pass `--artifacts C:\path\to\assets` to supported CLI commands and `-Artifacts` to the launcher to reuse an existing cache. Never commit model weights, datasets, uploads, indexes, or adapters.

```powershell
.venv\Scripts\python.exe -m sql_agent.cli --help
.venv\Scripts\python.exe -m sql_agent.cli seed --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli index-schema --artifacts artifacts --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli inspect-schema --artifacts artifacts --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli profile-values --artifacts artifacts --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli inspect-value-profile --artifacts artifacts --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli build-value-index --artifacts artifacts --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli inspect-evidence --artifacts artifacts --database artifacts\sql-agent\demo.sqlite --question "Which orders are from Singapore?"
.venv\Scripts\python.exe -m sql_agent.cli inspect-intent --artifacts artifacts --database artifacts\sql-agent\demo.sqlite --question "What is the total revenue from completed orders?"
.venv\Scripts\python.exe -m sql_agent.cli build-v8-knowledge --artifacts artifacts\v8 --database artifacts\sql-agent\demo.sqlite
.venv\Scripts\python.exe -m sql_agent.cli inspect-v8-evidence --artifacts artifacts\v8 --database artifacts\sql-agent\demo.sqlite --v8-evidence-mode all --question "Which orders are from Singapore?"
.venv\Scripts\python.exe -m sql_agent.cli inspect-v9-evidence --artifacts artifacts\v8 --database artifacts\sql-agent\demo.sqlite --question "Which orders are from Singapore?"
.venv\Scripts\python.exe -m sql_agent.cli verify-v9-evidence --artifacts artifacts\v8 --database artifacts\sql-agent\demo.sqlite --question "Which orders are from Singapore?"
.venv\Scripts\python.exe -m sql_agent.cli ask --artifacts artifacts --database artifacts\sql-agent\demo.sqlite --schema-mode auto --pipeline-profile v5 --question "What is the total revenue from completed orders?"
```

`auto` is the application default: it uses the full schema while the rendered schema fits the 10 KB prompt budget, then uses hybrid retrieval with foreign-key closure. Explicit hybrid mode does not silently fall back: a missing pinned BGE asset or schema index returns an actionable setup command. Cached schema and value indexes live outside source under `artifacts/sql-agent/`.

`pipeline_profile=v5` remains the evaluated public default. `v7_grounded` is available for local inspection and returns optional semantic intent, grounding, evidence, critic, result-signal, and repair metadata, but its pilot did not pass promotion gates. Public runtime permits at most two adaptive candidates; three-path generation and oracle evidence are evaluation-CLI-only.

Launch the interface with `.\scripts\run.ps1 -Role ui`; launch its API in another terminal with `.\scripts\run.ps1 -Role api`.

Seed a demonstration database with `python -m sql_agent.cli seed`. BIRD download is an explicit `python -m sql_agent.benchmark_setup` command and can require tens of gigabytes. API execution remains read-only; Mini-Dev uses the upstream set comparator with documented local guard limits, not an official leaderboard submission.

## Safety and publication

Bind services to localhost. Do not send private documents or traces to hosted services. Optional external tracing/judges remain opt-in and unverified. No paid compute, model publication or HR-policy compiler is included. Review all exported evidence and third-party licenses before publishing.

## Optional Docker path (not locally verified)

Build from this repository, not its parent. Copy the approved `portfolio_llm_evals-0.4.0-py3-none-any.whl` into `vendor/` first; the Dockerfile checks its SHA-256. Linux dependency resolution is separate from the tested Windows constraints. Do not describe this as a locked/tested Linux environment. Bind any published port explicitly to `127.0.0.1`; never expose the home Ollama service publicly.

## Reproduction details

[Complete commands and evaluation boundaries](docs/REPRODUCE.md).

[Architecture and trust boundaries](docs/ARCHITECTURE.md).


## V3 work

See [V3 correctness campaign](V3.md) for frozen evaluation roles, local commands and release boundaries.

