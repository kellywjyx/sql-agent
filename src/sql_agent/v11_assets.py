"""Explicit, checksum-recorded V11 model asset setup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v11_data import MODEL_REPOSITORY, MODEL_REVISION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(root: Path) -> Path:
    return root.resolve() / "v11/models/qwen2.5-coder-7b-instruct"


def download(root: Path) -> dict:
    from huggingface_hub import snapshot_download
    target = model_path(root)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPOSITORY, revision=MODEL_REVISION,
                      local_dir=target, local_dir_use_symlinks=False,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "LICENSE", "README.md"])
    files = {path.relative_to(target).as_posix(): _sha256(path)
             for path in sorted(target.rglob("*")) if path.is_file() and path.name != "asset-manifest.json"}
    manifest = {"version": "sql-v11-model-asset-v1", "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION, "license": "Apache-2.0", "files": files,
                "total_bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file())}
    (target / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(root: Path) -> dict:
    target = model_path(root); manifest_path = target / "asset-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("V11 model missing. Run `python scripts/download_sql_v11_model.py --artifacts <path>` after approving the 15.2 GB download.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != MODEL_REPOSITORY or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("V11 model repository or revision does not match the frozen protocol")
    for name, expected in manifest.get("files", {}).items():
        path = target / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"V11 model checksum mismatch: {name}")
    if not any(target.glob("*.safetensors")):
        raise RuntimeError("V11 model has no safetensors weights")
    return manifest
