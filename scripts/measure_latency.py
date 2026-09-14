"""Explicit cold/warm local Ollama measurements, separate from benchmark quality."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import httpx
from llm_evals.local import OllamaClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/latency"))
    args = parser.parse_args()
    target = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    for model in [args.model]:
        client = OllamaClient(model=model)
        identity = client.check()
        response = httpx.post(client.base_url + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        response.raise_for_status()
        for phase in ["after_explicit_unload", "warm_repeat"]:
            start = time.perf_counter()
            answer = client.chat([{"role": "user", "content": "Reply with the single word ready."}], max_tokens=16, think=False)
            rows.append({"model": identity, "phase": phase, "wall_seconds": time.perf_counter() - start,
                         "load_seconds": answer.load_seconds, "prompt_seconds": answer.prompt_seconds,
                         "generation_seconds": answer.generation_seconds, "output": answer.text})
        httpx.post(client.base_url + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60).raise_for_status()
    (target / "measurements.json").write_text(json.dumps({"cases": rows, "note": "Model residency probe; not app end-to-end latency or quality. Local API charge $0; electricity unmeasured."}, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
