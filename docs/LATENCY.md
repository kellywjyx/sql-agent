# Local model residency probe

Model: `llama3.1:8b`. Captured 28 August 2026 on the local 8 GB GPU.

| Phase | Wall seconds | Ollama load seconds |
|---|---:|---:|
| after_explicit_unload | 10.239 | 9.904 |
| warm_repeat | 0.246 | 0.001 |

One short text request after explicit unload, followed by one warm repeat. This is a model-residency diagnostic, not an application-latency benchmark, image-extraction measurement or repeated-trial estimate. End-to-end case latency is reported separately.

```powershell
.venv\Scripts\python.exe scripts/measure_latency.py --model llama3.1:8b
```

Run only when no other GPU jobs are active: this command explicitly unloads the selected Ollama model before and after measurement. It never downloads a model. Digests and raw timings are in [the measured JSON](latency-measurements.json). Local API charges are US$0; electricity and hardware costs were not measured.
