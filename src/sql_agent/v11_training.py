"""Strict CUDA-only NF4 QLoRA training for V11."""
from __future__ import annotations

import gc
import json
import math
import re
from pathlib import Path
from time import perf_counter

from llm_evals import load_cases
from llm_evals.campaign import immutable_json

from .v11_assets import model_path, verify
from .v11_data import MODEL_REVISION, open_role


TRAINING_VERSION = "sql-v11.2-qlora-v1"
# V11.2: covers the measured 2,900-token maximum; 2,048 truncated 10% of train and 28% of validation.
MAX_LENGTH = 3072
RECIPES = {"A": 2e-4, "B": 1e-4}


def environment() -> dict:
    import bitsandbytes
    import torch
    import transformers
    import trl
    import peft
    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    return {"cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "free_bytes": free, "total_bytes": total, "torch": torch.__version__,
            "transformers": transformers.__version__, "trl": trl.__version__,
            "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__}


def _load(root: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    verify(root); source = model_path(root)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
        device_map={"": 0}, quantization_config=quantization, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    devices = {str(parameter.device) for parameter in model.parameters()}
    if any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"V11 forbids CPU/disk offload; found parameter devices {sorted(devices)}")
    model.config.use_cache = False
    return model, tokenizer


def _dataset(cases):
    from datasets import Dataset
    return Dataset.from_list([{"prompt": case.context["messages"],
                               "completion": [{"role": "assistant", "content": case.expected["sql"]}]}
                              for case in cases])


def _trainer(model, tokenizer, train_cases, validation_cases, output: Path, *, learning_rate: float,
             gradient_accumulation: int, max_steps: int = -1, save_half: bool = True):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    optimizer_steps = max(1, math.ceil(len(train_cases) / gradient_accumulation))
    halfway = max(1, optimizer_steps // 2)
    args = SFTConfig(output_dir=str(output), seed=20260923, num_train_epochs=1,
        max_steps=max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation, learning_rate=learning_rate,
        max_length=MAX_LENGTH, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="paged_adamw_8bit",
        logging_steps=1 if max_steps > 0 else 5, logging_nan_inf_filter=False,
        eval_strategy="steps" if validation_cases and save_half else "no",
        eval_steps=halfway if validation_cases and save_half else None,
        save_strategy="steps" if save_half else "no", save_steps=halfway if save_half else None,
        save_total_limit=2, load_best_model_at_end=bool(validation_cases and save_half),
        metric_for_best_model="eval_loss" if validation_cases and save_half else None,
        greater_is_better=False if validation_cases and save_half else None,
        report_to="none", dataloader_num_workers=0,
        completion_only_loss=True, packing=False, warmup_ratio=.03)
    adapter = LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules="all-linear",
                         task_type="CAUSAL_LM", bias="none")
    return SFTTrainer(model=model, args=args, train_dataset=_dataset(train_cases),
                      eval_dataset=_dataset(validation_cases) if validation_cases else None,
                      peft_config=adapter, processing_class=tokenizer)


def preflight(root: Path, *, attempt_id: str = "attempt-1") -> dict:
    import torch
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", attempt_id):
        raise ValueError("V11 preflight attempt_id must be a safe lowercase identifier")
    attempt_root = root / "v11/sql-agent/preflight" / attempt_id
    if attempt_root.exists():
        raise ValueError(f"V11 preflight attempt {attempt_id!r} already exists; attempts are immutable")
    manifest = verify(root); hardware = environment()
    if not hardware["cuda_available"]:
        reason = "CUDA is unavailable; V11 stops without a smaller-model fallback"
        failure = {"version": TRAINING_VERSION, "status": "hardware_gate_failed", "reason": reason,
                   "hardware": hardware, "model_revision": manifest["revision"],
                   "gpu_seconds": 0, "training_started": False}
        target = attempt_root / "hardware-gate.json"
        immutable_json(target, {**failure, "attempt_id": attempt_id})
        raise RuntimeError(reason)
    if hardware["free_bytes"] < 7 * 1024**3:
        reason = "V11 requires at least 7 GiB free VRAM. Unload Ollama and close GPU-heavy applications."
        failure = {"version": TRAINING_VERSION, "status": "hardware_gate_failed", "reason": reason,
                   "hardware": hardware, "required_free_bytes": 7 * 1024**3,
                   "shortfall_bytes": 7 * 1024**3 - hardware["free_bytes"],
                   "model_revision": manifest["revision"], "gpu_seconds": 0,
                   "training_started": False}
        target = attempt_root / "hardware-gate.json"
        immutable_json(target, {**failure, "attempt_id": attempt_id})
        raise RuntimeError(reason)
    cases = open_role(root, "train")[:8]
    output = attempt_root / "adapter"
    model, tokenizer = _load(root)
    trainer = _trainer(model, tokenizer, cases, [], output, learning_rate=2e-4,
                       gradient_accumulation=8, max_steps=1, save_half=False)
    torch.cuda.reset_peak_memory_stats(); started = perf_counter()
    result = trainer.train(); elapsed = perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    if not math.isfinite(float(result.training_loss)):
        raise RuntimeError("V11 preflight produced non-finite loss")
    if peak > int(7.6 * 1024**3):
        raise RuntimeError(f"V11 preflight peak {peak} exceeds the 7.6 GiB gate")
    projected = elapsed * (1200 / 8)
    if projected > 2.2 * 3600:
        raise RuntimeError(f"Projected full training time {projected:.0f}s exceeds 2.2 hours")
    trainer.save_model(str(output)); tokenizer.save_pretrained(output)
    report = {"version": TRAINING_VERSION, "status": "trained; fresh-process reload pending",
              "model_asset": manifest, "hardware": hardware, "examples": 8,
              "training_seconds": elapsed, "projected_full_seconds": projected,
              "peak_gpu_bytes": peak, "training_loss": float(result.training_loss),
              "adapter_path": str(output), "no_offload": True}
    immutable_json(attempt_root / "training.json", {**report, "attempt_id": attempt_id})
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return report


def reload_check(root: Path, adapter: Path | None = None) -> dict:
    import torch
    from peft import PeftModel
    adapter = adapter or root / "v11/sql-agent/preflight/attempt-1/adapter"
    model, tokenizer = _load(root)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    case = open_role(root, "validation")[0]
    encoded = tokenizer.apply_chat_template(case.context["messages"], add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt", return_dict=True)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=32, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
    result = {"status": "reloaded", "adapter": str(adapter), "generated_nonempty": bool(generated.strip()),
              "sample": generated[:500], "model_revision": MODEL_REVISION}
    immutable_json(adapter.parent / "reload.json", result)
    return result


def train(root: Path, recipe: str, *, pilot: bool) -> dict:
    import torch
    if recipe not in RECIPES:
        raise ValueError("V11 recipe must be A or B")
    if not pilot:
        decision_path = root / "v11/sql-agent/pilot-decision.json"
        if not decision_path.is_file():
            raise RuntimeError("Freeze the V11 pilot decision before full training")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not decision.get("full_training_eligible") or decision.get("selected_recipe") != recipe:
            raise RuntimeError("Only the eligible frozen V11 pilot winner may enter full training")
    cases = open_role(root, "train")
    if pilot:
        cases = cases[:256]
    validation = open_role(root, "validation")
    output = root / f"v11/sql-agent/training/{'pilot' if pilot else 'full'}/{recipe}"
    if output.exists():
        raise ValueError("V11 training output already exists; immutable runs cannot be overwritten")
    model, tokenizer = _load(root); hardware = environment()
    trainer = _trainer(model, tokenizer, cases, validation, output,
                       learning_rate=RECIPES[recipe], gradient_accumulation=16)
    torch.cuda.reset_peak_memory_stats(); started = perf_counter(); result = trainer.train()
    elapsed = perf_counter() - started; history = trainer.state.log_history
    finite = math.isfinite(float(result.training_loss)) and all(
        math.isfinite(float(item[key])) for item in history for key in ("loss", "eval_loss", "grad_norm") if key in item)
    if not finite:
        raise RuntimeError("V11 training produced non-finite loss or gradients")
    final = output / "final"; trainer.save_model(str(final)); tokenizer.save_pretrained(final)
    report = {"version": TRAINING_VERSION, "recipe": recipe, "pilot": pilot,
              "learning_rate": RECIPES[recipe], "examples": len(cases), "validation_examples": len(validation),
              "training_seconds": elapsed, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "training_loss": float(result.training_loss), "loss_history": history,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "hardware": hardware, "adapter_path": str(final), "status": "trained; reload pending"}
    immutable_json(output / "training.json", report)
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return report
