"""Run one sequential V6 capture under the frozen four-hour ledger."""
import argparse
import sys
from pathlib import Path

from llm_evals import ComputeBudget


parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
parser.add_argument("--stage", choices=["pilot", "development", "final"], required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--estimate-seconds", type=float, required=True)
parser.add_argument("--reserve-seconds", type=float, default=0)
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
artifacts = args.artifacts.resolve()
budget = ComputeBudget(artifacts / "v6/compute", limits={"sql-agent": 14400}, total_seconds=14400)
command = [sys.executable, str(Path(__file__).with_name("run_sql_v6_eval.py")),
           "--artifacts", str(artifacts), "--stage", args.stage, "--mode", args.mode]
print(budget.run(args.job_id, "sql-agent", command, cwd=Path(__file__).resolve().parents[1],
                 estimate_seconds=args.estimate_seconds, reserve_seconds=args.reserve_seconds))
