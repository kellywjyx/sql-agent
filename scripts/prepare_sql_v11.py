import argparse
import json
from pathlib import Path

from sql_agent.v11_assets import model_path, verify
from sql_agent.v11_data import finalize, freeze_protocol

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
parser.add_argument("--protocol-only", action="store_true")
args = parser.parse_args(); root = args.artifacts.resolve()
if args.protocol_only:
    print(json.dumps(freeze_protocol(root), indent=2)); raise SystemExit(0)
verify(root)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path(root), local_files_only=True)
print(json.dumps(finalize(root, tokenizer, build_assets=True), indent=2))
