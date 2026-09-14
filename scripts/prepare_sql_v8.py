from pathlib import Path
import argparse
import json

from sql_agent.v8_data import prepare, prepare_pilot
from sql_agent.v8_decomposition import build

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
args = parser.parse_args()
data = prepare(args.artifacts)
decomposition = build(args.artifacts)
print(json.dumps({"data": data, "decomposition": decomposition,
                  "pilot": prepare_pilot(args.artifacts)}, indent=2))
