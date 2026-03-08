import re
from typing import Any
import pandas as pd
from sqlalchemy import text
import config
from config import (
    get_engine,
    FORBIDDEN_PATTERNS,
)

MAX_ROWS = 500
QUERY_TIMEOUT_SECONDS = 30
MAX_SCAN_ROWS = 1_000_000


class SQLExecutorAgent:
    def __init__(self):
        self.engine = get_engine()

    def explain(self, sql_query: str) -> dict[str, Any]:
        """Run EXPLAIN on a query and return plan text + warnings.

        Returns ``{"plan_text": str, "warnings": [str]}``
        """
        dialect = config.DB_DIALECT
        if dialect == "sqlite":
            explain_sql = f"EXPLAIN QUERY PLAN {sql_query}"
        else:
            explain_sql = f"EXPLAIN {sql_query}"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(explain_sql))
                rows = result.fetchall()

            lines = [" | ".join(str(v) for v in row) for row in rows]
            plan_text = "\n".join(lines)
            warnings = _parse_explain_warnings(plan_text, dialect)

            return {"plan_text": plan_text, "warnings": warnings}
        except Exception as exc:
            return {"plan_text": "", "warnings": [f"EXPLAIN failed: {exc}"]}

    def execute(self, sql_query: str) -> dict:
        """Execute a SQL query and return results.

        Returns dict with keys:
            success (bool), data (str), dataframe (pd.DataFrame | None), error (str | None)
        """
        if FORBIDDEN_PATTERNS.search(sql_query):
            return {
                "success": False,
                "data": "",
                "dataframe": None,
                "error": "Blocked: query contains forbidden statement. Only SELECT is allowed.",
            }

        try:
            with self.engine.connect() as conn:
                if config.DB_DIALECT != "sqlite":
                    conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME = {QUERY_TIMEOUT_SECONDS * 1000}"))

                result = conn.execute(text(sql_query))
                columns = list(result.keys())
                rows = result.fetchmany(MAX_ROWS)

            df = pd.DataFrame(rows, columns=columns)

            if df.empty:
                return {
                    "success": True,
                    "data": "Query returned no results.",
                    "dataframe": df,
                    "error": None,
                }

            formatted = df.to_string(index=False)
            return {
                "success": True,
                "data": formatted,
                "dataframe": df,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "data": "",
                "dataframe": None,
                "error": f"{type(e).__name__}: {e}",
            }


def _parse_explain_warnings(plan_text: str, dialect: str) -> list[str]:
    """Extract performance warnings from an EXPLAIN output, including row count guards."""
    warnings: list[str] = []
    lower = plan_text.lower()

    estimated_rows = _extract_estimated_rows(plan_text, dialect)

    if dialect == "sqlite":
        has_scan = "scan table" in lower or re.search(r"\bscan\s+\w+", lower)
        has_search = "search table" in lower or "search" in lower and "using" in lower
        if has_scan and not has_search:
            warnings.append("Full table scan detected (SCAN TABLE without index).")
        if "use temp b-tree" in lower:
            warnings.append("Temporary B-tree used for ORDER BY / GROUP BY (no covering index).")
    else:
        if "type: all" in lower or "\tall\t" in lower:
            warnings.append("Full table scan detected (type=ALL).")
        if "using filesort" in lower:
            warnings.append("Filesort detected -- consider adding an index.")
        if "using temporary" in lower:
            warnings.append("Temporary table used -- query may be expensive.")

    if estimated_rows > MAX_SCAN_ROWS:
        warnings.append(
            f"Query estimated to scan {estimated_rows:,} rows (threshold: {MAX_SCAN_ROWS:,}). "
            "Consider adding WHERE clauses or LIMIT to narrow the result set."
        )

    return warnings


def _extract_estimated_rows(plan_text: str, dialect: str) -> int:
    """Parse the EXPLAIN output to extract the maximum estimated row count."""
    max_rows = 0

    if dialect == "sqlite":
        for line in plan_text.split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                detail = parts[-1].lower()
                match = re.search(r"~(\d+)\s+rows", detail)
                if match:
                    max_rows = max(max_rows, int(match.group(1)))
    else:
        for line in plan_text.split("\n"):
            match = re.search(r"\brows[:\s]*(\d+)", line, re.IGNORECASE)
            if match:
                max_rows = max(max_rows, int(match.group(1)))

    return max_rows
