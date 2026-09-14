"""Record/verify exact BIRD database bytes before and after a local campaign."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from llm_evals import load_cases


def inventory(root):
    cases = [case for split in ["development", "final"] for case in load_cases(root / f"{split}.jsonl")]
    records = {}
    for path in sorted({Path(c.context["database"]).resolve(strict=True) for c in cases}):
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Benchmark database escaped the selected artifact directory")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        records[path.relative_to(root.resolve()).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["record", "verify"])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    target = args.artifacts.resolve() / "v2/sql-agent/bird"
    path = target / "database-bytes-before.json"
    if args.command == "record" and path.exists():
        raise ValueError("Before-run inventory already exists; do not overwrite it")
    current = inventory(target)
    if args.command == "record":
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"recorded_at": datetime.now(timezone.utc).isoformat(), "databases": current}, handle, indent=2)
        print(f"Recorded {len(current)} benchmark database hashes before model evaluation")
    else:
        previous = json.loads(path.read_text(encoding="utf-8"))["databases"]
        if current != previous:
            raise ValueError("Benchmark database bytes changed; read-only verification failed")
        output = target / ("database-bytes-verified-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
        with output.open("x", encoding="utf-8") as handle:
            json.dump({"verified_at": datetime.now(timezone.utc).isoformat(), "unchanged": True,
                       "database_count": len(current), "before_inventory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, handle, indent=2)
        print(f"Verified unchanged bytes for {len(current)} benchmark databases")
