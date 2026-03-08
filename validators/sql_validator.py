from __future__ import annotations
from typing import Any
import sqlglot
from sqlglot import exp
from config import FORBIDDEN_PATTERNS

_DIALECT_MAP = {
    "sqlite": "sqlite",
    "mysql": "mysql",
}


def validate_sql(sql: str, dialect: str, schema_meta: dict[str, list[str]]) -> dict[str, Any]:
    """Validate a SQL string against the known schema.

    Args:
        sql: The SQL query string.
        dialect: DB dialect key ("sqlite" or "mysql").
        schema_meta: ``{table_name: [col1, col2, ...]}`` (all lower-cased).

    Returns:
        ``{"valid": bool, "errors": [str], "tables_used": [str], "columns_used": [str]}``
    """
    errors: list[str] = []

    if FORBIDDEN_PATTERNS.search(sql):
        return {
            "valid": False,
            "errors": ["Query contains forbidden write statement (only SELECT allowed)."],
            "tables_used": [],
            "columns_used": [],
        }

    sg_dialect = _DIALECT_MAP.get(dialect, "sqlite")

    try:
        parsed = sqlglot.parse_one(sql, dialect=sg_dialect)
    except sqlglot.errors.ParseError as exc:
        return {
            "valid": False,
            "errors": [f"Syntax error: {exc}"],
            "tables_used": [],
            "columns_used": [],
        }

    dialect_errors = _check_dialect_functions(parsed, dialect)
    errors.extend(dialect_errors)

    cte_names = _extract_cte_names(parsed)
    subquery_aliases = _extract_subquery_aliases(parsed)
    virtual_tables = cte_names | subquery_aliases

    all_tables = _extract_tables(parsed)
    real_tables = all_tables - virtual_tables
    columns_used = _extract_columns(parsed)

    known_tables = set(schema_meta.keys())
    unknown_tables = real_tables - known_tables
    if unknown_tables:
        errors.append(f"Unknown table(s): {', '.join(sorted(unknown_tables))}")

    query_aliases = _extract_aliases(parsed)
    all_known_columns: set[str] = set()
    for cols in schema_meta.values():
        all_known_columns.update(cols)
    all_known_columns.update(query_aliases)

    for col in sorted(columns_used):
        if col == "*":
            continue
        if col not in all_known_columns:
            errors.append(f"Unknown column: {col}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "tables_used": sorted(real_tables),
        "columns_used": sorted(columns_used - {"*"}),
    }


def _extract_cte_names(tree: exp.Expression) -> set[str]:
    """Collect names defined by WITH clauses (CTEs)."""
    names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            names.add(alias.lower())
    return names


def _extract_subquery_aliases(tree: exp.Expression) -> set[str]:
    """Collect aliases of subqueries used as derived tables."""
    names: set[str] = set()
    for subq in tree.find_all(exp.Subquery):
        alias = subq.alias
        if alias:
            names.add(alias.lower())
    return names


def _extract_aliases(tree: exp.Expression) -> set[str]:
    """Collect all column-level aliases defined anywhere in the query."""
    aliases: set[str] = set()
    for node in tree.find_all(exp.Alias):
        alias = node.alias
        if alias:
            aliases.add(alias.lower())
    return aliases


def _extract_tables(tree: exp.Expression) -> set[str]:
    """Pull all table names referenced in the AST."""
    tables: set[str] = set()
    for node in tree.find_all(exp.Table):
        name = node.name
        if name:
            tables.add(name.lower())
    return tables


def _extract_columns(tree: exp.Expression) -> set[str]:
    """Pull all column names referenced in the AST (ignoring aliases)."""
    columns: set[str] = set()
    for node in tree.find_all(exp.Column):
        name = node.name
        if name:
            columns.add(name.lower())
    return columns


_MYSQL_FORBIDDEN_FUNCS = {"strftime"}
_SQLITE_FORBIDDEN_FUNCS = {"year", "month", "day", "hour", "minute", "second"}


def _check_dialect_functions(tree: exp.Expression, dialect: str) -> list[str]:
    """Flag functions that belong to the wrong dialect."""
    forbidden = (
        _MYSQL_FORBIDDEN_FUNCS if dialect == "mysql" else
        _SQLITE_FORBIDDEN_FUNCS if dialect == "sqlite" else
        set()
    )
    if not forbidden:
        return []
    errors: list[str] = []
    for func in tree.find_all(exp.Anonymous, exp.Func):
        name = ""
        if isinstance(func, exp.Anonymous):
            name = func.name.lower() if func.name else ""
        elif hasattr(func, "sql_name"):
            name = func.sql_name().lower()
        if name in forbidden:
            if dialect == "mysql":
                errors.append(
                    f"strftime() does not exist in MySQL. Use YEAR(), MONTH(), etc. instead."
                )
            else:
                errors.append(
                    f"{name}() does not exist in SQLite. Use strftime() instead."
                )
            break
    return errors
