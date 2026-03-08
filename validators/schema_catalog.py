from __future__ import annotations
import os
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import (
    get_engine,
    get_schema_meta,
    get_fk_graph,
    get_table_row_counts,
    require_openai_key,
)

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "chroma_store")
COLLECTION_NAME = "schema_tables"


class SchemaCatalog:
    """Persistent, embedding-indexed catalog of table descriptions."""

    def __init__(self, chroma_dir: str = CHROMA_DIR):
        self._chroma_dir = os.path.abspath(chroma_dir)
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            ef = OpenAIEmbeddingFunction(
                api_key=require_openai_key(),
                model_name="text-embedding-3-small",
            )
            client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ensure_built(self, engine=None) -> None:
        """Build the catalog if it hasn't been populated yet."""
        collection = self._get_collection()
        if collection.count() > 0:
            return
        self._rebuild(engine)

    def _rebuild(self, engine=None) -> None:
        if engine is None:
            engine = get_engine()

        schema_meta = get_schema_meta(engine)
        fk_graph = get_fk_graph(engine)
        row_counts = get_table_row_counts(engine)

        ids, docs, metas = [], [], []
        for table, columns in schema_meta.items():
            desc = _build_description(table, columns, fk_graph.get(table, []), row_counts.get(table, 0))
            ids.append(table)
            docs.append(desc)
            metas.append({"table_name": table, "column_count": len(columns), "row_count": row_counts.get(table, 0)})

        if ids:
            collection = self._get_collection()
            collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def rebuild(self, engine=None) -> None:
        """Force a full rebuild of the catalog."""
        collection = self._get_collection()
        existing = collection.get()["ids"]
        if existing:
            collection.delete(ids=existing)
        self._rebuild(engine)

    def search_tables(self, question: str, top_k: int = 5) -> list[str]:
        """Return the most relevant table names for a question."""
        self.ensure_built()
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        return [meta["table_name"] for meta in results["metadatas"][0]]

    def get_description(self, table_name: str) -> Optional[str]:
        """Return the stored description for a single table."""
        self.ensure_built()
        collection = self._get_collection()
        result = collection.get(ids=[table_name.lower()], include=["documents"])
        if result["documents"]:
            return result["documents"][0]
        return None

    def get_all_descriptions(self) -> dict[str, str]:
        """Return {table_name: description} for all tables."""
        self.ensure_built()
        collection = self._get_collection()
        result = collection.get(include=["documents", "metadatas"])
        out = {}
        for doc, meta in zip(result["documents"], result["metadatas"]):
            out[meta["table_name"]] = doc
        return out


def _build_description(
    table: str,
    columns: list[str],
    fk_edges: list[dict],
    row_count: int,
) -> str:
    """Generate a natural-language description of a table for embedding."""
    col_str = ", ".join(columns)
    parts = [f"{table}: columns {col_str}."]
    if row_count > 0:
        parts.append(f"~{row_count} rows.")
    if fk_edges:
        fk_str = "; ".join(f"{e['column']} -> {e['references']}" for e in fk_edges)
        parts.append(f"FK: {fk_str}.")
    else:
        parts.append("No foreign keys.")
    return " ".join(parts)
