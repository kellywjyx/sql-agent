from pathlib import Path
import argparse
import json

from sql_agent.q5_diagnostic import report

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
print(json.dumps(report(args.artifacts), indent=2))
