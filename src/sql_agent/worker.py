"""Disposable subprocess: read-only database, authorizer, VM and wall-clock budgets."""
import json
from pathlib import Path
import sqlite3
import sys
import time

from .guardrails import authorizer, validate
from .policy import POLICY


def execute(request: dict) -> dict:
    sql = validate(request["sql"])
    path = Path(request["database"]).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.5)
    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 20000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, POLICY["schema_column_limit"])
        connection.set_authorizer(authorizer)
        deadline = time.monotonic() + request.get("timeout", 3)
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = connection.execute(sql)
        if len(cursor.description or []) > POLICY["result_column_limit"]:
            raise ValueError("Result exceeds the independent 64-column returned-result limit")
        limit = request.get("max_rows", 10000)
        safe_rows, byte_count = [], 0
        for row in cursor:
            if len(safe_rows) >= limit:
                raise ValueError(f"Result exceeds {limit} rows; refine the query. Evaluation is not truncated.")
            row = [{"blob_hex": v.hex()} if isinstance(v, bytes) else v for v in row]
            byte_count += len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
            if byte_count > 2 * 1024 * 1024:
                raise ValueError("Result exceeds the 2 MiB output budget")
            safe_rows.append(row)
        return {"sql": sql, "columns": [c[0] for c in cursor.description], "rows": safe_rows, "row_count": len(safe_rows),
                "execution_policy": POLICY["id"]}
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        result = execute(json.load(sys.stdin))
        print(json.dumps({"ok": True, **result}, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
