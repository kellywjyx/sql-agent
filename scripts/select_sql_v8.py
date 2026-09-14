import argparse
import json
from pathlib import Path

from sql_agent.v8_selection import freeze, freeze_development

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path("../artifacts"))
parser.add_argument("--stage", choices=["pilot", "development"], default="pilot")
args = parser.parse_args()
print(json.dumps(freeze(args.artifacts) if args.stage == "pilot" else freeze_development(args.artifacts), indent=2))
