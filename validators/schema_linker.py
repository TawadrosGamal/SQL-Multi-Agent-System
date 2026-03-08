from __future__ import annotations
import re
from config import (
    get_schema_meta,
    get_schema_info,
    get_schema_info_for_tables,
)
from validators.schema_catalog import SchemaCatalog
from validators.table_selector import select_tables

SMALL_SCHEMA_THRESHOLD = 10
EMBEDDING_TOP_K = 10
PROGRESSIVE_THRESHOLD = 5


def link_schema(
    question: str,
    engine=None,
    *,
    threshold: int = SMALL_SCHEMA_THRESHOLD,
) -> tuple[str, list[str]]:
    """Return (filtered_schema_info, list_of_linked_table_names).

    If the schema is small enough the full schema is returned unchanged.
    For larger schemas:
      1. Embedding search via SchemaCatalog (top-K)
      2. Keyword boost to re-rank and catch exact-name matches
      3. If > PROGRESSIVE_THRESHOLD candidates remain, call the LLM table
         selector to narrow down further
    """
    schema_meta = get_schema_meta(engine)

    if len(schema_meta) <= threshold:
        all_tables = sorted(schema_meta.keys())
        return get_schema_info(engine), all_tables

    candidates = _hybrid_search(question, schema_meta, engine)

    if len(candidates) > PROGRESSIVE_THRESHOLD:
        candidates = _progressive_disclosure(question, candidates)

    if not candidates:
        candidates = sorted(schema_meta.keys())

    schema_str = get_schema_info_for_tables(candidates, engine)
    return schema_str, candidates


def _hybrid_search(
    question: str,
    schema_meta: dict[str, list[str]],
    engine=None,
) -> list[str]:
    """Combine embedding search with keyword scoring."""
    catalog = SchemaCatalog()
    catalog.ensure_built(engine)

    embedding_results = catalog.search_tables(question, top_k=EMBEDDING_TOP_K)

    tokens = _tokenize(question)
    scored: dict[str, float] = {}

    for i, table in enumerate(embedding_results):
        scored[table] = 10.0 - i * 0.5

    for table, columns in schema_meta.items():
        kw = _keyword_score(tokens, table, columns)
        if kw > 0:
            scored[table] = scored.get(table, 0) + kw

    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [t for t, s in ranked if s > 0]


def _progressive_disclosure(question: str, candidates: list[str]) -> list[str]:
    """Use a lightweight LLM call to narrow down the candidate set."""

    catalog = SchemaCatalog()
    descs = catalog.get_all_descriptions()

    candidate_descs = {t: descs.get(t, t) for t in candidates}
    try:
        selected = select_tables(question, candidate_descs, max_tables=PROGRESSIVE_THRESHOLD)
        return selected if selected else candidates[:PROGRESSIVE_THRESHOLD]
    except Exception:
        return candidates[:PROGRESSIVE_THRESHOLD]


def _tokenize(text: str) -> list[str]:
    """Split question into lowercase keyword tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    words = text.split()
    stop = {"the", "a", "an", "in", "of", "for", "and", "or", "by", "to",
            "is", "are", "what", "which", "how", "show", "me", "get", "find",
            "each", "per", "all", "with", "from", "that", "this", "top"}
    return [w for w in words if w not in stop and len(w) > 1]


def _keyword_score(tokens: list[str], table: str, columns: list[str]) -> float:
    """Score a table's relevance to the question based on keyword overlap."""
    score = 0.0
    table_parts = set(table.lower().replace("_", " ").split())
    col_set = {c.lower() for c in columns}
    col_parts: set[str] = set()
    for c in columns:
        col_parts.update(c.lower().replace("_", " ").split())

    for token in tokens:
        if token in table_parts:
            score += 3.0
        if token in col_set:
            score += 2.0
        elif token in col_parts:
            score += 1.0
        if token in table.lower():
            score += 0.5

    return score
