import json
from pathlib import Path

from sql_agent.v10_memory import build

root = Path(__file__).resolve().parents[2] / "artifacts"
print(json.dumps(build(root, force=True), indent=2))
