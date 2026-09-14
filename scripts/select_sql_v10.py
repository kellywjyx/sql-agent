import json
from pathlib import Path

from sql_agent.v10_selection import freeze

root = Path(__file__).resolve().parents[2] / "artifacts"
print(json.dumps(freeze(root), indent=2))
