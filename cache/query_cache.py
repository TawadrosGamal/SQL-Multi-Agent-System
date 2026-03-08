
import hashlib
import os
import re
import sqlite3
import time
from typing import Optional
from config import require_openai_key, get_schema_meta
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

CACHE_DB = os.path.join(os.path.dirname(__file__), "..", "pipeline_cache.db")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "query_cache"

STRICT_SIMILARITY = 0.88
RELAXED_SIMILARITY = 0.78
KEYWORD_BOOST_THRESHOLD = 0.35

_STOP_WORDS = frozenset(
    "the a an in of for and or by to is are what which how show me get "
    "find each per all with from that this top can you list give".split()
)


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def _hash(q: str) -> str:
    return hashlib.sha256(_norm(q).encode()).hexdigest()


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer for cache keyword matching."""
    for suffix in ("ing", "tion", "ment", "ness", "able", "ible", "ies", "ed", "ly", "er", "est", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _keywords(text: str) -> set[str]:
    """Extract stemmed keyword tokens from a question."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {_stem(w) for w in words - _STOP_WORDS}


_ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}

_ORDINAL_SUFFIX = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def _extract_numbers(text: str) -> set[str]:
    """Pull digit sequences, ordinal words, and suffixed ordinals (1st, 2nd, 3rd)."""
    nums = set(re.findall(r"\d+", text))
    for tok in text.lower().split():
        if tok in _ORDINAL_WORDS:
            nums.add(_ORDINAL_WORDS[tok])
    for m in _ORDINAL_SUFFIX.finditer(text):
        nums.add(m.group(1))
    return nums


def _keyword_overlap(q1: str, q2: str) -> float:
    """Jaccard similarity of keyword sets (0.0 to 1.0)."""
    k1, k2 = _keywords(q1), _keywords(q2)
    if not k1 or not k2:
        return 0.0
    return len(k1 & k2) / len(k1 | k2)


_entity_vocab: set[str] | None = None


def _build_entity_vocab() -> set[str]:
    """Derive entity vocabulary from the live DB schema.
    Splits table and column names on underscores, stems each token,
    and returns the set of meaningful entity words (e.g. "order",
    "customer", "product", "category", "region", ...).
    """
    meta = get_schema_meta()
    vocab: set[str] = set()
    for table, columns in meta.items():
        for part in table.split("_"):
            if len(part) > 1:
                vocab.add(_stem(part))
        for col in columns:
            for part in col.split("_"):
                if len(part) > 1:
                    vocab.add(_stem(part))
    return vocab


def _get_entity_vocab() -> set[str]:
    """Return the cached entity vocabulary, building it on first access."""
    global _entity_vocab
    if _entity_vocab is None:
        _entity_vocab = _build_entity_vocab()
    return _entity_vocab


def _reset_entity_vocab() -> None:
    """Clear the cached entity vocabulary so it rebuilds from the new schema."""
    global _entity_vocab
    _entity_vocab = None


def _has_entity_swap(q1: str, q2: str) -> bool:
    """Detect when two queries differ by swapping DB entity nouns.

    Uses the actual database schema to identify entity words (table names,
    column name stems). Only flags a swap when both sides of the symmetric
    difference contribute different DB entity words -- e.g. "order" on one
    side and "customer" on the other.

    Non-entity words (verbs, adjectives, structural words) are ignored
    automatically since they won't appear in the schema-derived vocabulary.
    """
    w1 = {_stem(w) for w in re.findall(r"[a-z]+", q1.lower()) if w not in _STOP_WORDS}
    w2 = {_stem(w) for w in re.findall(r"[a-z]+", q2.lower()) if w not in _STOP_WORDS}
    only1 = w1 - w2
    only2 = w2 - w1
    if not only1 or not only2:
        return False
    entity_vocab = _get_entity_vocab()
    entities1 = only1 & entity_vocab
    entities2 = only2 & entity_vocab
    return bool(entities1) and bool(entities2)


class QueryCache:
    def __init__(self, db_path: str = CACHE_DB, chroma_dir: str = CHROMA_DIR):
        self.db_path = os.path.abspath(db_path)
        self._chroma_dir = os.path.abspath(chroma_dir)
        self._init_db()
        self._collection = None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exact_cache (
                    question_hash TEXT PRIMARY KEY,
                    question TEXT,
                    sql_query TEXT,
                    query_results TEXT,
                    summary TEXT,
                    created_at REAL
                )
            """)

    def _get_collection(self):
        if self._collection is None:
            # To use the free local model instead of OpenAI, replace the
            # embedding_function with chromadb's default:
            #   client.get_or_create_collection(COLLECTION_NAME)
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

    def get_exact(self, question: str) -> Optional[dict]:
        h = _hash(question)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM exact_cache WHERE question_hash = ?", (h,)
            ).fetchone()
        if row:
            return dict(row)
        return None

    def put_exact(self, question: str, sql_query: str, query_results: str, summary: str):
        h = _hash(question)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO exact_cache
                   (question_hash, question, sql_query, query_results, summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (h, _norm(question), sql_query, query_results, summary, time.time()),
            )

    def get_semantic(self, question: str) -> Optional[dict]:
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return None

            results = collection.query(
                query_texts=[question],
                n_results=1,
                include=["metadatas", "distances", "documents"],
            )

            if not results["ids"] or not results["ids"][0]:
                return None

            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            matched_doc = results["documents"][0][0] if results["documents"] and results["documents"][0] else ""

            if _extract_numbers(question) != _extract_numbers(matched_doc):
                return None

            if _has_entity_swap(question, matched_doc):
                return None

            kw_overlap = _keyword_overlap(question, matched_doc)
            effective_threshold = (
                RELAXED_SIMILARITY if kw_overlap >= KEYWORD_BOOST_THRESHOLD
                else STRICT_SIMILARITY
            )
            is_hit = similarity >= effective_threshold

            if not is_hit:
                return None

            meta = results["metadatas"][0][0]
            return {
                "question": matched_doc,
                "sql_query": meta.get("sql_query", ""),
                "query_results": meta.get("query_results", ""),
                "summary": meta.get("summary", ""),
                "similarity": similarity,
            }
        except Exception:
            return None

    def put_semantic(self, question: str, sql_query: str, query_results: str, summary: str):
        try:
            collection = self._get_collection()
            doc_id = _hash(question)

            collection.upsert(
                ids=[doc_id],
                documents=[question],
                metadatas=[{
                    "sql_query": sql_query,
                    "query_results": query_results[:4000],
                    "summary": summary,
                    "created_at": str(time.time()),
                }],
            )
        except Exception:
            pass

    def get(self, question: str) -> Optional[dict]:
        """Try exact match first, then semantic similarity."""
        hit = self.get_exact(question)
        if hit:
            hit["cache_type"] = "exact"
            return hit

        hit = self.get_semantic(question)
        if hit:
            hit["cache_type"] = "semantic"
            return hit

        return None

    def put(self, question: str, sql_query: str, query_results: str, summary: str):
        """Store in both caches."""
        self.put_exact(question, sql_query, query_results, summary)
        self.put_semantic(question, sql_query, query_results, summary)

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM exact_cache")
        try:
            collection = self._get_collection()
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
        except Exception:
            pass
