import argparse
import sys
from pathlib import Path

from llm_evals import ComputeBudget

parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
parser.add_argument("--kind", choices=["preflight", "reload", "pilot-train", "full-train", "evaluate"], required=True)
parser.add_argument("--recipe", choices=["A", "B"], default="A")
parser.add_argument("--stage", choices=["pilot", "development", "regression"], default="pilot")
parser.add_argument("--variant", choices=["base", "A", "B", "selected"], default="base")
parser.add_argument("--estimate-seconds", type=float, required=True)
parser.add_argument("--reserve-seconds", type=float, default=0)
parser.add_argument("--attempt-id", default="attempt-1")
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args(); root = args.artifacts.resolve(); repository = Path(__file__).resolve().parents[1]
if args.kind in {"preflight", "reload", "pilot-train", "full-train"}:
    command_name = {"preflight": "preflight", "reload": "reload", "pilot-train": "pilot", "full-train": "full"}[args.kind]
    command = [sys.executable, str(Path(__file__).with_name("train_sql_v11.py")), command_name,
               "--recipe", args.recipe, "--artifacts", str(root)]
    if args.kind == "preflight": command.extend(["--attempt-id", args.attempt_id])
else:
    command = [sys.executable, str(Path(__file__).with_name("run_sql_v11_eval.py")),
               "--artifacts", str(root), "--stage", args.stage, "--variant", args.variant]
budget = ComputeBudget(root / "v11/compute", limits={"sql-agent": 21600}, total_seconds=21600)
print(budget.run(args.job_id, "sql-agent", command, cwd=repository,
                 estimate_seconds=args.estimate_seconds, reserve_seconds=args.reserve_seconds))
