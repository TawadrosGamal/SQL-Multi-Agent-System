import os
import re
import argparse
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_DEFAULT_DB = "sqlite:///" + os.path.join(os.path.dirname(__file__), "retail_orders.db")
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DB)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
DB_DIALECT = "mysql" if "mysql" in DATABASE_URL else "sqlite"

FORBIDDEN_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o":        {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":   {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50,  "output": 1.50},
}


def compute_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost given token counts and model name."""
    fallback = {"input": 0.15, "output": 0.60}
    rates = MODEL_COSTS.get(model, fallback)
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000

_DB_PRESETS: dict[str, str] = {}

for _key, _val in os.environ.items():
    if _key.startswith("DB_") and _key not in ("DB_DIALECT",) and "://" in str(_val):
        _label = _key[3:].replace("_", " ").title()
        _DB_PRESETS[_label] = _val

if not _DB_PRESETS:
    _DB_PRESETS["Sqlite"] = _DEFAULT_DB


def get_db_presets() -> dict[str, str]:
    """Return ``{label: url}`` for every configured database preset."""
    return dict(_DB_PRESETS)


def set_database(url: str) -> None:
    """Switch the active database at runtime.

    Updates the module-level ``DATABASE_URL`` and ``DB_DIALECT`` so that
    subsequent calls to ``get_engine()`` reflect the new target.
    """
    global DATABASE_URL, DB_DIALECT
    DATABASE_URL = url
    DB_DIALECT = "mysql" if "mysql" in url else "sqlite"


def select_db_from_args(description: str = "") -> str:
    """Parse ``--db`` from the command line and switch the active database.

    Accepts a preset label (e.g. ``sqlite``, ``mysql``) matched
    case-insensitively against ``DB_*`` env vars, or a raw connection URL.
    Returns the active ``DATABASE_URL`` after any switch.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--db",
        default=None,
        help="DB preset label (e.g. sqlite, mysql) or a full connection URL",
    )
    args, _ = parser.parse_known_args()
    if args.db:
        presets = get_db_presets()
        match = next(
            (url for label, url in presets.items()
             if label.lower() == args.db.lower()),
            None,
        )
        if match:
            set_database(match)
        elif "://" in args.db:
            set_database(args.db)
        else:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Unknown DB preset '{args.db}'. Available: {available}"
            )
    return DATABASE_URL


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Create a .env file from .env.example.")
    return OPENAI_API_KEY


def get_engine():
    return create_engine(DATABASE_URL)


def get_schema_info(engine=None) -> str:
    """Dynamically introspect the database and return a schema description string."""
    if engine is None:
        engine = get_engine()

    inspector = inspect(engine)
    lines = []

    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_defs = ", ".join(f"{c['name']} {c['type']}" for c in columns)
        lines.append(f"Table: {table_name} ({col_defs})")

        pk = inspector.get_pk_constraint(table_name)
        if pk and pk.get("constrained_columns"):
            lines.append(f"  Primary key: {', '.join(pk['constrained_columns'])}")

    if not lines:
        return "No tables found in the database. Run setup_db.py first."

    return "\n".join(lines)


def get_schema_meta(engine=None) -> dict:
    """Return {table_name: [column_name, ...]} for every table in the DB."""
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    meta = {}
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        meta[table_name.lower()] = [c["name"].lower() for c in columns]
    return meta


def get_schema_info_for_tables(tables: list[str], engine=None) -> str:
    """Return schema description for only the specified tables."""
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    lines = []
    for table_name in inspector.get_table_names():
        if table_name.lower() not in {t.lower() for t in tables}:
            continue
        columns = inspector.get_columns(table_name)
        col_defs = ", ".join(f"{c['name']} {c['type']}" for c in columns)
        lines.append(f"Table: {table_name} ({col_defs})")
        pk = inspector.get_pk_constraint(table_name)
        if pk and pk.get("constrained_columns"):
            lines.append(f"  Primary key: {', '.join(pk['constrained_columns'])}")
    if not lines:
        return "No matching tables found."
    return "\n".join(lines)


def get_fk_graph(engine=None) -> dict[str, list[dict]]:
    """Return foreign key relationships for every table.

    Returns ``{"orders": [{"column": "product_id", "references": "products.id"}], ...}``
    Tables with no FKs map to an empty list.
    """
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    graph: dict[str, list[dict]] = {}
    for table_name in inspector.get_table_names():
        fks = inspector.get_foreign_keys(table_name)
        edges = []
        for fk in fks:
            ref_table = fk.get("referred_table", "")
            for local_col, ref_col in zip(
                fk.get("constrained_columns", []),
                fk.get("referred_columns", []),
            ):
                edges.append({
                    "column": local_col,
                    "references": f"{ref_table}.{ref_col}",
                })
        graph[table_name.lower()] = edges
    return graph


def get_table_row_counts(engine=None) -> dict[str, int]:
    """Return approximate row counts for every table."""
    if engine is None:
        engine = get_engine()
    counts: dict[str, int] = {}
    inspector = inspect(engine)
    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            try:
                row = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                counts[table_name.lower()] = row[0] if row else 0
            except Exception:
                counts[table_name.lower()] = -1
    return counts


_DATE_COL_PATTERNS = re.compile(r"(date|_at$|_on$|timestamp)", re.IGNORECASE)


def get_column_samples(
    engine=None,
    tables: list[str] | None = None,
    max_distinct: int = 20,
) -> dict[str, list[str]]:
    """Return distinct values for low-cardinality columns.

    Only includes columns with <= *max_distinct* unique values.
    Also extracts distinct years from date-like columns regardless of cardinality.
    Returns ``{"orders.region": ["Central","East","South","West"], ...}``
    """
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    samples: dict[str, list[str]] = {}
    target_tables = {t.lower() for t in tables} if tables else None
    dialect = DB_DIALECT

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            if target_tables and table_name.lower() not in target_tables:
                continue
            columns = inspector.get_columns(table_name)
            for col in columns:
                col_name = col["name"]
                col_type = str(col.get("type", "")).lower()
                is_date_col = (
                    "date" in col_type or "time" in col_type
                    or _DATE_COL_PATTERNS.search(col_name)
                )
                if is_date_col:
                    _add_year_samples(conn, table_name, col_name, dialect, samples)
                    continue
                try:
                    row = conn.execute(
                        text(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
                    ).fetchone()
                    n_distinct = row[0] if row else 0
                    if 2 <= n_distinct <= max_distinct:
                        rows = conn.execute(
                            text(f"SELECT DISTINCT {col_name} FROM {table_name} ORDER BY {col_name}")
                        ).fetchall()
                        vals = [str(r[0]) for r in rows if r[0] is not None]
                        if vals:
                            samples[f"{table_name.lower()}.{col_name.lower()}"] = vals
                except Exception:
                    continue
    return samples


def _add_year_samples(conn, table_name: str, col_name: str, dialect: str, samples: dict):
    """Extract distinct years from a date column and add to samples dict."""
    try:
        if dialect == "sqlite":
            year_sql = f"SELECT DISTINCT strftime('%Y', {col_name}) AS yr FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY yr"
        else:
            year_sql = f"SELECT DISTINCT YEAR({col_name}) AS yr FROM {table_name} WHERE {col_name} IS NOT NULL ORDER BY yr"
        rows = conn.execute(text(year_sql)).fetchall()
        years = [str(r[0]) for r in rows if r[0] is not None]
        if years:
            samples[f"{table_name.lower()}.{col_name.lower()}_year"] = years
    except Exception:
        pass


def format_column_hints(samples: dict[str, list[str]]) -> str:
    """Format column samples into a string for the LLM prompt."""
    if not samples:
        return ""
    lines = []
    for col_key, values in sorted(samples.items()):
        lines.append(f"  {col_key} has values: {', '.join(values)}")
    return "COLUMN VALUE HINTS:\n" + "\n".join(lines)


def format_fk_info(fk_graph: dict[str, list[dict]], tables: list[str] | None = None) -> str:
    """Format FK relationships into a string for the LLM prompt."""
    lines = []
    for table, edges in sorted(fk_graph.items()):
        if tables and table not in {t.lower() for t in tables}:
            continue
        for edge in edges:
            lines.append(f"  {table}.{edge['column']} -> {edge['references']}")
    if not lines:
        return ""
    return "FOREIGN KEY RELATIONSHIPS:\n" + "\n".join(lines)


def get_sample_rows(engine=None, table_name: str = "orders", limit: int = 3) -> str:
    """Return a few sample rows so the LLM understands the data."""
    if engine is None:
        engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
        columns = list(result.keys())
        rows = result.fetchall()

    header = " | ".join(columns)
    separator = " | ".join("---" for _ in columns)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
    return f"{header}\n{separator}\n{body}"
