from pathlib import Path

from sql_agent.schema import inspect_schema
from sql_agent.v10_data import open_memory_capture, prepare_memory_capture
from sql_agent.v8_knowledge import KnowledgeCache

root = Path(__file__).resolve().parents[2] / "artifacts"
prepare_memory_capture(root)
cases = open_memory_capture(root)
cache = KnowledgeCache(root / "v10")
for database in sorted({Path(case.context["database"]) for case in cases}):
    manifest = cache.build(database, inspect_schema(database))
    print(database.name, manifest["identity"][:12], manifest["version"])
