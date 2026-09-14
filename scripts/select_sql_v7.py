from pathlib import Path
import argparse
import json

from sql_agent.v7_selection import freeze, freeze_pilot_decision, freeze_reproduction_decision

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
parser.add_argument("--stage", choices=["reproduction", "pilot", "selection"], default="selection")
args = parser.parse_args()
functions = {"reproduction": freeze_reproduction_decision, "pilot": freeze_pilot_decision,
             "selection": freeze}
print(json.dumps(functions[args.stage](args.artifacts), indent=2))
