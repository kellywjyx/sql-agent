"""Record the explicitly approved Arctic Q5 asset with remote and local checksums."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from llm_evals.campaign import immutable_json


REPOSITORY = "mradermacher/Arctic-Text2SQL-R1-7B-GGUF"
FILENAME = "Arctic-Text2SQL-R1-7B.Q5_K_M.gguf"
UPSTREAM = "Snowflake/Arctic-Text2SQL-R1-7B"
OLLAMA_NAME = "hf.co/mradermacher/Arctic-Text2SQL-R1-7B-GGUF:Q5_K_M"


def ollama_json(path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:11434{path}", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
parser.add_argument("--approved", action="store_true")
args = parser.parse_args()
if not args.approved:
    parser.error("Q5 provenance recording requires the user's explicit --approved acknowledgement")

api = HfApi()
quant = api.model_info(REPOSITORY)
metadata = get_hf_file_metadata(hf_hub_url(REPOSITORY, FILENAME, revision=quant.sha))
upstream = api.model_info(UPSTREAM)
tags = ollama_json("/api/tags").get("models", [])
installed = next((row for row in tags if row.get("name") == OLLAMA_NAME), None)
if installed is None:
    raise RuntimeError(f"Approved model is not installed. Run: ollama pull {OLLAMA_NAME}")
show = ollama_json("/api/show", {"name": OLLAMA_NAME})
from_match = re.search(r"^FROM\s+(.+)$", show.get("modelfile", ""), flags=re.MULTILINE)
match = re.search(r"sha256-([0-9a-f]{64})", from_match.group(1) if from_match else "")
if not from_match or not match:
    raise RuntimeError("Could not resolve the local Q5 GGUF blob from Ollama")
blob_path = Path(from_match.group(1).strip())
blob_sha = match.group(1)
remote_sha = str(metadata.etag).strip('"').removeprefix("sha256:")
if remote_sha != blob_sha:
    raise RuntimeError(f"Q5 local/remote GGUF checksum mismatch: {blob_sha} != {remote_sha}")
if not blob_path.is_file() or int(metadata.size or 0) != blob_path.stat().st_size:
    raise RuntimeError("Q5 local/remote byte size mismatch")

card_data = upstream.card_data.to_dict() if upstream.card_data is not None else {}
license_name = card_data.get("license")
record = {
    "schema_version": 1,
    "approval_scope": "20-case matched Q5 quantization diagnostic only",
    "ollama_name": OLLAMA_NAME,
    "ollama_digest": installed["digest"],
    "ollama_size": installed["size"],
    "quant_repository": REPOSITORY,
    "quant_repository_revision": quant.sha,
    "gguf_file": FILENAME,
    "gguf_sha256": blob_sha,
    "gguf_bytes": metadata.size,
    "upstream_repository": UPSTREAM,
    "upstream_revision": upstream.sha,
    "license": license_name,
    "api_cost_usd": 0,
    "locked_final_opened": False,
}
target = args.artifacts.resolve() / "v7/sql-agent/models/arctic-q5.json"
immutable_json(target, record)
print(json.dumps(record, indent=2))
