"""Run one sequential V7 model capture under the frozen six-hour ledger."""
import argparse
import sys
from pathlib import Path

from llm_evals import ComputeBudget

parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
parser.add_argument("--stage", choices=["reproduction", "calibration", "selection", "regression"], required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--estimate-seconds", type=float, required=True)
parser.add_argument("--reserve-seconds", type=float, default=0)
parser.add_argument("--pilot-size", type=int)
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
artifacts = args.artifacts.resolve()
budget = ComputeBudget(artifacts / "v7/compute", limits={"sql-agent": 21600}, total_seconds=21600)
command = [sys.executable, str(Path(__file__).with_name("run_sql_v7_eval.py")),
           "--artifacts", str(artifacts), "--stage", args.stage, "--mode", args.mode]
if args.pilot_size:
    command.extend(["--pilot-size", str(args.pilot_size)])
print(budget.run(args.job_id, "sql-agent", command, cwd=Path(__file__).resolve().parents[1],
                 estimate_seconds=args.estimate_seconds, reserve_seconds=args.reserve_seconds))
