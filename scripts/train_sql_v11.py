import argparse
import json
from pathlib import Path

from sql_agent.v11_training import preflight, reload_check, train

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["preflight", "reload", "pilot", "full"])
parser.add_argument("--recipe", choices=["A", "B"], default="A")
parser.add_argument("--adapter", type=Path)
parser.add_argument("--attempt-id", default="attempt-1")
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args(); root = args.artifacts.resolve()
if args.command == "preflight": result = preflight(root, attempt_id=args.attempt_id)
elif args.command == "reload": result = reload_check(root, args.adapter)
else: result = train(root, args.recipe, pilot=args.command == "pilot")
print(json.dumps(result, indent=2))
