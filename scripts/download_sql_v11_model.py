import argparse
import json
from pathlib import Path

from sql_agent.v11_assets import download

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
print(json.dumps(download(args.artifacts.resolve()), indent=2))
