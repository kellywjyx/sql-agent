import argparse
import sys
from pathlib import Path

from llm_evals import ComputeBudget

parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
parser.add_argument("--estimate-seconds", type=float, default=500)
parser.add_argument("--reserve-seconds", type=float, default=0)
args = parser.parse_args()
artifacts = Path(__file__).resolve().parents[2] / "artifacts"
budget = ComputeBudget(artifacts / "v10/compute", limits={"sql-agent": 3600}, total_seconds=3600)
command = [sys.executable, str(Path(__file__).with_name("run_sql_v10_qwen.py"))]
print(budget.run(args.job_id, "sql-agent", command, cwd=Path(__file__).resolve().parents[1],
                 estimate_seconds=args.estimate_seconds, reserve_seconds=args.reserve_seconds))
