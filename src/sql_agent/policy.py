"""Versioned production policy, shared by applications and benchmark workers."""
POLICY = {
    "id": "sqlite-readonly-v3", "schema_column_limit": 256,
    "result_column_limit": 64, "max_rows": 10000, "max_result_bytes": 2 * 1024 * 1024,
    "max_sql_bytes": 20000, "current_timestamp": "denied", "default_timeout_seconds": 3,
    "read_only": True, "extension_loading": False,
}
