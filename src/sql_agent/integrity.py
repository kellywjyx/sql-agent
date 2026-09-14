"""Content hashes used to prove evaluation databases remain unchanged."""
import hashlib
from pathlib import Path


def snapshot_database_hashes(cases):
    paths = sorted({Path(case.context["database"]).resolve() for case in cases})
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def changed_databases(before):
    return [Path(path).name for path, digest in before.items()
            if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
