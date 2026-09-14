# V11 results: stopped at hardware preflight

## Outcome

V11 did not enter training. The approved protocol says that failure of the 7B preflight ends the campaign without changing models, offloading weights, or weakening memory safeguards.

| Check | Measured result | Gate | Outcome |
|---|---:|---:|---|
| Model revision | `dcc04bb77301e6368aa24a609c6ea496b70503e8` | Exact match | Pass |
| Snapshot size | 15,242,808,321 bytes | Checksum manifest | Pass |
| Frozen role sizes | 1,200 / 150 / 200 | Exact | Pass |
| Prompt token coverage | 1.000 / 1.000 / 1.000 | At least 0.85 | Pass |
| Human BIRD evidence in inputs | False | False | Pass |
| CUDA runtime | PyTorch 2.8.0+cu128 | CUDA required | Pass |
| Free VRAM | 7,406,092,288 bytes | At least 7,516,192,768 | **Fail** |
| VRAM shortfall | 110,100,480 bytes | 0 | **Fail** |
| Training started | False | Only after all preflight gates | Correct stop |
| GPU training time | 0 seconds | Six-hour ceiling | Within budget |
| API cost | $0 | $0 | Pass |
| Locked final opened | False | False | Pass |

The GPU was an NVIDIA GeForce RTX 4060 Ti. Ollama reported no loaded models. The remaining device memory was insufficient for the explicit 7 GiB pre-load requirement, so the model was never loaded and the eight-example forward/backward step was not attempted.

## What can and cannot be concluded

The implementation, dataset isolation, prompt serialization, model provenance, CUDA environment, and safety-preserving interfaces were verified. The experiment does **not** establish that QLoRA would improve or fail to improve SQL execution accuracy. There are no V11 training losses, adapters, base-versus-adapter predictions, or EX results.

Qwen V5 remains the public default with its recorded 0.545 execution accuracy. V11 is a measured local-hardware limitation rather than a model-quality result. A future retry requires a separate decision about freeing additional VRAM or using different hardware; it must not be represented as part of this completed run.

## Verification

- 100 offline tests passed after V11 integration and immutable retry support.
- The pinned 15.24 GB model asset passed full SHA-256 verification.
- All 58 selected-database knowledge caches were created; 60 column profiles are explicitly marked partial after bounded query timeouts.
- No SQL evaluation ran, so database post-evaluation hashes are not applicable.
- The compute ledger records the preflight command failure, and `hardware-gate.json` records that training never started.
