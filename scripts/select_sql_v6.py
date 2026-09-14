import argparse
import json
from pathlib import Path

from sql_agent.v6_reporting import final_report
from sql_agent.v6_selection import freeze_selection, pilot_gate


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["pilot-gate", "freeze-selection", "final-report"])
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
functions = {"pilot-gate": pilot_gate, "freeze-selection": freeze_selection, "final-report": final_report}
print(json.dumps(functions[args.action](args.artifacts.resolve()), indent=2))
