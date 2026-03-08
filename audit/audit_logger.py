from __future__ import annotations
import json
import os
import sqlite3
import time
from typing import Any, Optional

AUDIT_DB = os.path.join(os.path.dirname(__file__), "..", "audit_log.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    question        TEXT    NOT NULL,
    linked_tables   TEXT,
    sql_query       TEXT,
    validation_ok   INTEGER,
    validation_errors TEXT,
    explain_plan    TEXT,
    explain_warnings TEXT,
    execution_success INTEGER,
    query_results_preview TEXT,
    summary         TEXT,
    cache_hit       TEXT,
    elapsed_seconds REAL,
    error           TEXT
)
"""


class AuditLogger:
    def __init__(self, db_path: str = AUDIT_DB):
        self._db_path = os.path.abspath(db_path)
        self._ensure_table()

    def _ensure_table(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_TABLE)

    def log(self, entry: dict[str, Any]) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO audit_log
                   (timestamp, question, linked_tables, sql_query,
                    validation_ok, validation_errors,
                    explain_plan, explain_warnings,
                    execution_success, query_results_preview,
                    summary, cache_hit, elapsed_seconds, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    entry.get("question", ""),
                    _to_json(entry.get("linked_tables")),
                    entry.get("sql_query", ""),
                    1 if entry.get("validation_ok") else 0,
                    _to_json(entry.get("validation_errors")),
                    entry.get("explain_plan", ""),
                    _to_json(entry.get("explain_warnings")),
                    1 if entry.get("execution_success") else 0,
                    (entry.get("query_results", "") or "")[:500],
                    entry.get("summary", ""),
                    entry.get("cache_hit"),
                    entry.get("elapsed_seconds"),
                    entry.get("error"),
                ),
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM audit_log")


def _to_json(val: Any) -> Optional[str]:
    if val is None:
        return None
    return json.dumps(val) if not isinstance(val, str) else val
