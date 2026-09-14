import json
from pathlib import Path
import subprocess
import sys
import os

from llm_evals.tracing import traced

from .guardrails import validate


@traced
def execute_sql(database: Path, sql: str, *, timeout: float = 3, max_rows: int = 10000) -> dict:
    validate(sql)
    request = {"database": str(database.resolve()), "sql": sql, "timeout": timeout, "max_rows": max_rows}
    try:
        process = subprocess.run([sys.executable, "-m", "sql_agent.worker"], input=json.dumps(request),
                                 capture_output=True, text=True, encoding="utf-8", timeout=timeout + 3,
                                 env={**os.environ, "PYTHONUTF8": "1"})
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Query worker exceeded its wall-clock deadline and was terminated") from exc
    if process.returncode:
        raise RuntimeError(f"Query worker failed: {process.stderr[:500]}")
    result = json.loads(process.stdout)
    if not result["ok"]:
        raise ValueError(result["error"])
    return result
