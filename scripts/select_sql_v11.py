import argparse
import json
from pathlib import Path

from sql_agent.v11_selection import freeze_development, freeze_pilot, freeze_regression

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["pilot", "development", "regression"], required=True)
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args(); root = args.artifacts.resolve()
function = {"pilot": freeze_pilot, "development": freeze_development, "regression": freeze_regression}[args.stage]
print(json.dumps(function(root), indent=2))
