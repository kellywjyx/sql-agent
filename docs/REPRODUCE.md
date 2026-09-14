# Reproduce sql-agent

Run the repository README's independent setup first, from this repository directory. The commands below do not depend on parent scripts. Downloads occur only in explicit preparation commands. GPU model jobs must run sequentially. Do not open or tune against final examples before freezing development decisions.

```powershell
.venv\Scripts\python.exe -m pip install -c requirements.lock.txt -e '.[benchmarks]'
ollama pull llama3.1:8b
.venv\Scripts\python.exe -m sql_agent.cli seed
.venv\Scripts\python.exe -m sql_agent.cli evaluate --ablate
# Explicit large download: about 9.3 GB compressed plus extracted databases.
.venv\Scripts\python.exe -m sql_agent.benchmark_setup --output artifacts/v2/sql-agent/bird
.venv\Scripts\python.exe scripts/audit_databases.py record
.venv\Scripts\python.exe -m sql_agent.campaign --split development
# Freeze decisions before opening Mini-Dev.
.venv\Scripts\python.exe -m sql_agent.campaign --split final
.venv\Scripts\python.exe scripts/audit_databases.py verify
.venv\Scripts\python.exe scripts/report_comparison.py --split development
.venv\Scripts\python.exe scripts/report_comparison.py --split final
.venv\Scripts\python.exe scripts/analyze_failures.py --split development
.venv\Scripts\python.exe scripts/analyze_failures.py --split final
```

Default assets and reports live under `artifacts/`, excluded from Git. Use the same external asset root throughout when moving assets. The demonstration CLI accepts `--database` and `--output`; benchmark commands accept `--artifacts`.

Reports are immutable `<output>/<run-id>/` directories; `latest.json` is a navigation pointer. Preserve failed runs. Compare dataset content identities and metric versions, not paths or timestamps. Legacy v1 receipt scores and path-based identities cannot be compared directly with v2.

Model download revisions and weight checksums are packaged in llm-evals resources. BIRD archive/scorer revisions are packaged in sql-agent; sentiment revisions in fin-adapter. Ollama tags may change: record the installed digest from `/api/tags`, and do not call a different digest an exact reproduction. The upstream FastAPI commit is checked exactly.

Offline tests cover implementation behavior and security. Local model results, hosted CI runs and Docker runtime checks are separate evidence. Do not configure the home GPU as an untrusted pull-request runner.

## V7 exposed-development campaign

V7 never opens the 1,033-case locked V6 final. The following commands reproduce
the frozen exposed roles and local artifacts. Run model jobs sequentially.

```powershell
.venv\Scripts\python.exe scripts\prepare_sql_v7.py --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\run_sql_v7_eval.py --artifacts ..\artifacts --stage reproduction --mode reference_generated --prepare-assets
.venv\Scripts\python.exe scripts\run_sql_v7_job.py --artifacts ..\artifacts --job-id v7-reference-generated --stage reproduction --mode reference_generated --estimate-seconds 500
.venv\Scripts\python.exe scripts\run_sql_v7_job.py --artifacts ..\artifacts --job-id v7-three-pilot --stage selection --mode v7_three --pilot-size 20 --estimate-seconds 750
.venv\Scripts\python.exe scripts\select_sql_v7.py --artifacts ..\artifacts --stage reproduction
.venv\Scripts\python.exe scripts\select_sql_v7.py --artifacts ..\artifacts --stage pilot
```

The measured pilot decision is `continue_full_selection: false`; do not run the
60-case calibration, 100-case selection, regression replay, Talon experiment, or
locked final as though V7 passed. A separately approved Q5 diagnostic was later
run on exactly 20 matched oracle-evidence cases:

```powershell
ollama pull hf.co/mradermacher/Arctic-Text2SQL-R1-7B-GGUF:Q5_K_M
.venv\Scripts\python.exe scripts\record_sql_v7_q5_asset.py --artifacts ..\artifacts --approved
.venv\Scripts\python.exe scripts\run_sql_v7_job.py --artifacts ..\artifacts --job-id v7-q5-matched-oracle-20 --stage reproduction --mode reference_q5_oracle --pilot-size 20 --estimate-seconds 360 --reserve-seconds 18000
.venv\Scripts\python.exe scripts\report_sql_v7_q5.py --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\report_sql_v7_q5_cases.py --artifacts ..\artifacts
```

Q5 scored 0.500 versus 0.550 for matched Q4, with zero recoveries and one
regression. Do not extend this into Q6/Q8/BF16 or Talon work. The Q5 profile is
evaluation-only and rejects any pilot size other than 20.

## V8 exposed evidence-reconstruction campaign

V8 reuses only the 40 already exposed V7 reproduction cases and Arctic Q4. Build
the CPU caches and evaluate the scoring-only retrieval gate first:

```powershell
.venv\Scripts\python.exe scripts\prepare_sql_v8.py --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\run_sql_v8_evidence.py --artifacts ..\artifacts
```

Gate B must pass before any model job. Run the four 20-case conditions
sequentially under the one-hour V8 ledger, then freeze the pilot decision:

```powershell
.venv\Scripts\python.exe scripts\run_sql_v8_job.py --job-id v8-pilot-literals-01 --mode generated_literals --estimate-seconds 600 --reserve-seconds 1800 --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\run_sql_v8_job.py --job-id v8-pilot-descriptions-01 --mode generated_descriptions --estimate-seconds 600 --reserve-seconds 1200 --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\run_sql_v8_job.py --job-id v8-pilot-normalization-01 --mode generated_normalization --estimate-seconds 600 --reserve-seconds 600 --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\run_sql_v8_job.py --job-id v8-pilot-all-01 --mode generated_all --estimate-seconds 600 --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\select_sql_v8.py --stage pilot --artifacts ..\artifacts
```

Only the frozen winner may run on all 40 exposed cases:

```powershell
.venv\Scripts\python.exe scripts\run_sql_v8_job.py --job-id v8-development-all-01 --mode generated_all --scope development --estimate-seconds 900 --artifacts ..\artifacts
.venv\Scripts\python.exe scripts\select_sql_v8.py --stage development --artifacts ..\artifacts
```

The recorded development gate is false. Do not open the locked final, add Talon,
or represent V8 as promoted. See `docs/V8-RESULTS.md`.

## V9 execution-grounded semantic reasoning

V9 reuses only the exact exposed V8 cases. The preparation step freezes the
protocol, builds cache-v2 through read-only connections, and reproduces the
five-recovery/three-regression audit:

```powershell
.venv\Scripts\python.exe scripts\prepare_sql_v9.py
.venv\Scripts\python.exe scripts\run_sql_v9_job.py --job-id v9-p0-cache-v2 --mode p0_cache_v2 --scope pilot --estimate-seconds 450 --reserve-seconds 1000
.venv\Scripts\python.exe scripts\run_sql_v9_job.py --job-id v9-b-scoped --mode scoped --scope pilot --estimate-seconds 450 --reserve-seconds 500
.venv\Scripts\python.exe scripts\select_sql_v9.py
```

The immutable scoping decision permits the probe stage. Run it once, then freeze
the final pilot decision:

```powershell
.venv\Scripts\python.exe scripts\run_sql_v9_job.py --job-id v9-c-probed --mode probed --scope pilot --estimate-seconds 500 --reserve-seconds 100
.venv\Scripts\python.exe -c "from pathlib import Path; from sql_agent.v9_selection import freeze_pilot; freeze_pilot(Path('../artifacts'))"
```

The recorded `development_eligible` value is false. Do not run a development or
locked-final job, and do not describe V9 as promoted. See `docs/V9-RESULTS.md`.

## V10 error-conditioned semantic correction

V10 freezes a fresh 40-case exposed pilot before building its correction memory.
Gold SQL is used only by the offline mutation builder, calibration scorer, and EX
callbacks; it is never included in application requests or correction records.

```powershell
.venv\Scripts\python.exe scripts\prepare_sql_v10.py
.venv\Scripts\python.exe scripts\build_sql_v10_knowledge.py
.venv\Scripts\python.exe scripts\build_sql_v10_memory.py
.venv\Scripts\python.exe scripts\run_sql_v10_job.py --job-id v10-memory-arctic-30 --scope memory --estimate-seconds 400 --reserve-seconds 1500
.venv\Scripts\python.exe scripts\build_sql_v10_natural_memory.py
.venv\Scripts\python.exe scripts\calibrate_sql_v10.py
```

After calibration is frozen, run the fresh Arctic control, deterministic replay,
and same-set Qwen V5 control sequentially:

```powershell
.venv\Scripts\python.exe scripts\run_sql_v10_job.py --job-id v10-pilot-scoped-control-40 --scope pilot --estimate-seconds 650 --reserve-seconds 900
.venv\Scripts\python.exe scripts\run_sql_v10_replay.py
.venv\Scripts\python.exe scripts\run_sql_v10_qwen_job.py --job-id v10-pilot-qwen-v5-control-40 --estimate-seconds 500
.venv\Scripts\python.exe scripts\select_sql_v10.py
```

The recorded decision has `gate_passed: false` and
`selective_qwen_correction_eligible: false`. Do not run a selective model
corrector, broader campaign, or locked-final evaluation as though V10 passed.
See `docs/V10-RESULTS.md`.

## V11 direct QLoRA adaptation

V11 uses a separate Python 3.13 training environment. Model weights, datasets,
indexes, and adapters remain outside Git. First install the checksum-pinned
evaluation wheel and exact CUDA training stack, then download and freeze the
pinned model and data:

```powershell
.\scripts\setup-training.ps1 -EvalsWheel ..\artifacts\releases\v0.4.0\wheels\portfolio_llm_evals-0.4.0-py3-none-any.whl -EvalsSHA256 3f0f45d9c16bf8f7af773cd19bd0425fc2c77b61a74f86ef4f9c7ffbf2bd3b30
.venv\Scripts\python.exe scripts\download_sql_v11_model.py --artifacts ..\artifacts --approved
.venv\Scripts\python.exe scripts\prepare_sql_v11.py --artifacts ..\artifacts
.venv\Scripts\python.exe -m pytest -q
```

Unload Ollama models and run the strict preflight through the six-hour compute
ledger:

```powershell
ollama ps
.venv-train\Scripts\python.exe scripts\run_sql_v11_job.py --job-id v11-preflight-20260902 --kind preflight --recipe A --estimate-seconds 1440 --artifacts ..\artifacts
```

The recorded run stopped before model load because 7,406,092,288 free bytes were
below the 7 GiB requirement by 110,100,480 bytes. Do not run pilot training,
development evaluation, inspected regression, or the locked final as though the
preflight passed. See `docs/V11-RESULTS.md`.

A separately authorized retry must use a new immutable attempt and job identity;
it must not overwrite the measured failure:

```powershell
.venv-train\Scripts\python.exe scripts\run_sql_v11_job.py --job-id v11-preflight-attempt-2 --kind preflight --attempt-id attempt-2 --recipe A --estimate-seconds 1440 --artifacts ..\artifacts
```
