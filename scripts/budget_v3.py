"""One immediate local v3 job under the shared eight-hour compute ledger."""
import argparse
import os
from pathlib import Path
import sys
from llm_evals.budget import ComputeBudget

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--estimate", type=float, required=True)
parser.add_argument("--reserve", type=float, default=0)
args = parser.parse_args()
root = args.artifacts.resolve()
project = "sql-agent"
command = [sys.executable, "-m", project.replace("-", "_") + ".v3_campaign", "--artifacts", str(root), "--stage", args.stage, "--mode", args.mode]
ComputeBudget(root / "v3/compute").run(f"{project}-{args.stage}-{args.mode}", project, command,
    cwd=Path(__file__).resolve().parents[1], estimate_seconds=args.estimate, reserve_seconds=args.reserve,
    env={"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(root / "hf-cache")})
