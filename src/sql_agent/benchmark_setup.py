"""Explicit BIRD downloads. Never execute downloaded code or SQL during setup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import zipfile

from llm_evals.dataset import write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target):
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + ".partial")
    # curl supplies a whole-transfer deadline and low-speed cutoff in addition
    # to connect/read timeouts; partial archives remain resumable after failure.
    result = subprocess.run(["curl.exe" if __import__("os").name == "nt" else "curl", "--fail", "--location",
                             "--continue-at", "-", "--max-time", "1800", "--speed-time", "60", "--speed-limit", "1024",
                             "--retry", "2", "--retry-delay", "2", "--max-filesize", str(45 * 1024**3),
                             "--output", str(partial), url], timeout=5500)
    if result.returncode:
        raise RuntimeError(f"Download failed ({result.returncode}); retained resumable partial: {partial}")
    partial.replace(target)


def extract(archive, target):
    target = target.resolve()
    with zipfile.ZipFile(archive) as handle:
        if sum(item.file_size for item in handle.infolist()) > 80 * 1024**3:
            raise ValueError("Archive expansion exceeds 80 GiB")
        for item in handle.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.orig_filename or ":" in item.filename:
                raise ValueError("Unsafe archive path")
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue  # AppleDouble metadata is not a nested archive.
            destination = (target / item.filename).resolve()
            if not destination.is_relative_to(target) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive escape or symlink")
            if item.is_dir() or destination.suffix.lower() not in {".sqlite", ".json", ".csv", ".zip", ".txt"}:
                continue
            if destination.exists() and destination.stat().st_size == item.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(item) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)


def prepare(target):
    from huggingface_hub import hf_hub_download
    pins = json.loads((Path(__file__).parent / "resources/bird-sources.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    manifest_file = target / "manifest.json"
    if manifest_file.exists():
        print("BIRD setup already complete; manifests retained")
        return
    downloads = target / "downloads"
    downloads.mkdir(exist_ok=True)
    provenance = {}
    for split in ["train", "dev"]:
        url = f"https://bird-bench.oss-cn-beijing.aliyuncs.com/{split}.zip"
        archive = downloads / f"{split}.zip"
        print(f"Downloading official BIRD {split} archive", flush=True)
        download(url, archive)
        if sha256(archive) != pins["source"][split]["sha256"]:
            raise ValueError(f"Pinned BIRD {split} checksum mismatch; do not evaluate changed data")
        provenance[split] = {"url": url, "sha256": sha256(archive), "bytes": archive.stat().st_size}
        extract(archive, target / split)
        for nested in (target / split).rglob("*.zip"):
            if "__MACOSX" in nested.parts or nested.name.startswith("._"):
                continue
            extract(nested, nested.parent / nested.stem)
    repository = "birdsql/bird_mini_dev"
    revision = pins["mini_dev"]["revision"]
    source = hf_hub_download(repository, "data/mini_dev_sqlite-00000-of-00001.json", repo_type="dataset", revision=revision,
                            local_dir=str(downloads / "mini-dev"), cache_dir=str(downloads / "hf-cache"))
    if sha256(Path(source)) != pins["mini_dev"]["sha256"]:
        raise ValueError("Pinned Mini-Dev checksum mismatch")
    text = Path(source).read_text(encoding="utf-8")
    records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(records) != 500:
        raise ValueError("Expected the approved 500-case SELECT-only Mini-Dev version")
    training_files = list((target / "train").rglob("train.json"))
    if len(training_files) != 1:
        raise ValueError("Expected exactly one upstream train.json")
    training = json.loads(training_files[0].read_text(encoding="utf-8"))
    selected = sorted(random.Random(42).sample(range(len(training)), 100))
    from .evaluation import load_benchmark
    for name, rows, split in [("development", [training[i] for i in selected], "train"), ("final", records, "dev")]:
        database_files = [path for path in (target / split).rglob("*.sqlite")
                          if "__MACOSX" not in path.parts and not path.name.startswith("._")]
        database_roots = {path.parent.parent for path in database_files}
        if len(database_roots) != 1:
            raise ValueError(f"Ambiguous {split} database root; expected db_id/db_id.sqlite layout")
        data_file = target / f"{name}.json"
        write_json(data_file, rows)
        cases = load_benchmark(data_file, next(iter(database_roots)), "bird-mini-dev" if name == "final" else "bird-train")
        (target / f"{name}.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8")
    commit = pins["scorer"]["commit"]
    scorer_url = f"https://raw.githubusercontent.com/bird-bench/mini_dev/{commit}/evaluation/evaluation_ex.py"
    scorer = downloads / "evaluation_ex.py"
    download(scorer_url, scorer)
    if sha256(scorer) != pins["scorer"]["sha256"]:
        raise ValueError("Pinned scorer checksum mismatch")
    write_json(manifest_file, {"source": provenance, "mini_dev": {"repository": repository, "revision": revision,
               "sha256": sha256(Path(source))}, "scorer": {"url": scorer_url, "commit": commit, "sha256": sha256(scorer)},
               "development_train_indices": selected, "counts": {"development": 100, "final": 500},
               "license": "BIRD CC BY-SA 4.0; retain upstream attribution", "downloaded_code_executed": False})
    print("Prepared BIRD train=100, Mini-Dev=500; no benchmark inference executed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2/sql-agent/bird"))
    prepare(parser.parse_args().output.resolve())
