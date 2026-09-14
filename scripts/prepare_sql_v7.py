from pathlib import Path
import argparse
import json

from sql_agent.v7_data import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
print(json.dumps(prepare(args.artifacts), indent=2))
