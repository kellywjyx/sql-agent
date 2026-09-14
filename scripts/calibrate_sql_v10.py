import json
from pathlib import Path

from sql_agent.v10_calibration import calibrate

root = Path(__file__).resolve().parents[2] / "artifacts"
print(json.dumps(calibrate(root), indent=2))
