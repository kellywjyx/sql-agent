import json
from pathlib import Path

from sql_agent.schema import inspect_schema
from sql_agent.v8_data import open_cases
from sql_agent.v8_knowledge import KnowledgeCache
from sql_agent.v9_audit import build
from sql_agent.v9_data import prepare

root = Path(__file__).resolve().parents[2] / "artifacts"
print(json.dumps(prepare(root), indent=2))
cache = KnowledgeCache(root / "v8")
for database in sorted({Path(case.context["database"]) for case in open_cases(root)}):
    manifest = cache.build(database, inspect_schema(database))
    print(database.name, manifest["version"], manifest["identity"][:12])
print(json.dumps(build(root), indent=2))
