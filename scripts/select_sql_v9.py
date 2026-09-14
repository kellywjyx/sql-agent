import json
from pathlib import Path

from sql_agent.v9_selection import freeze_pilot, freeze_scoping

root = Path(__file__).resolve().parents[2] / "artifacts"
scoping = freeze_scoping(root)
print(json.dumps(scoping, indent=2))
if not scoping["probes_eligible"]:
    print(json.dumps(freeze_pilot(root), indent=2))
