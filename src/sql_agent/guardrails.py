from __future__ import annotations

import sqlite3

import sqlglot
from sqlglot import exp


class UnsafeSQL(ValueError):
    pass


FUNCTIONS = {"abs", "avg", "coalesce", "count", "date", "datetime", "glob", "group_concat", "hex",
             "ifnull", "iif", "instr", "julianday", "length", "like", "likelihood", "lower", "ltrim",
             "max", "min", "nullif", "replace", "round", "rtrim", "strftime", "substr", "substring",
             "sum", "time", "total", "trim", "typeof", "unicode", "unixepoch", "upper",
             "row_number", "rank", "dense_rank", "lag", "lead", "first_value", "last_value", "ntile"}


def validate(sql: str) -> str:
    if len(sql) > 20000:
        raise UnsafeSQL("SQL exceeds length limit")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        raise UnsafeSQL(f"SQL parse error: {exc}") from exc
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeSQL("Only a single SELECT query is allowed")
    tree = statements[0]
    forbidden_names = {"Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Pragma",
                       "Attach", "Detach", "Copy", "Merge", "Transaction", "Into", "Lock", "Use", "Grant"}
    for node in tree.walk():
        if type(node).__name__ in forbidden_names:
            raise UnsafeSQL("Unsupported or mutating SQL construct")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise UnsafeSQL("Recursive CTEs are disabled")
        if isinstance(node, exp.Table):
            if not isinstance(node.this, exp.Identifier) or node.catalog or node.db:
                raise UnsafeSQL("Only unqualified, registered database tables are allowed")
            if node.name.casefold().startswith(("sqlite_", "pragma_")):
                raise UnsafeSQL("Internal catalog and PRAGMA table access is disabled")
        if isinstance(node, exp.Anonymous) and node.name.lower() not in FUNCTIONS:
            raise UnsafeSQL(f"Function not allowlisted: {node.name}")
    return tree.sql(dialect="sqlite")


def authorizer(action, arg1, arg2, database, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # SQLite reports database=None, column='' for COUNT(*) table reads.
        known_database = database == "main" or (database is None and arg2 == "")
        return sqlite3.SQLITE_OK if known_database and not (arg1 or "").lower().startswith("sqlite_") else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or arg1 or "").lower() in FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY
