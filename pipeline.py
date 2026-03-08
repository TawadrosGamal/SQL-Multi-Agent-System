from __future__ import annotations
import time
import config
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from config import (
    get_engine,
    get_schema_meta,
    get_sample_rows,
    get_fk_graph,
    get_column_samples,
    format_fk_info,
    format_column_hints,
    compute_llm_cost,
    MODEL_NAME,
)
from agents.sql_generator import SQLGeneratorAgent
from agents.sql_executor import SQLExecutorAgent
from agents.summarizer import SummarizerAgent
from cache.query_cache import QueryCache
from validators.sql_validator import validate_sql as _validate_sql_fn
from validators.schema_linker import link_schema as _link_schema_fn
from audit.audit_logger import AuditLogger
import pandas as pd
from sqlalchemy import text as sa_text
from cache.query_cache import _reset_entity_vocab
MAX_RETRIES = 2


class PipelineState(TypedDict):
    question: str
    schema_info: str
    sample_rows: str
    sql_query: str
    query_results: str
    dataframe: Optional[object]
    summary: str
    error: Optional[str]
    retry_count: int
    # cache
    cache_hit: Optional[str]  # "exact", "semantic", or None
    use_cache: bool
    # schema linking
    linked_tables: Optional[list[str]]
    # validation
    validation_ok: Optional[bool]
    validation_errors: Optional[list[str]]
    # explain
    explain_plan: Optional[str]
    explain_warnings: Optional[list[str]]
    # enriched context
    fk_info: Optional[str]
    column_hints: Optional[str]
    # timing
    _start_time: Optional[float]
    # LLM cost tracking
    llm_usage: Optional[dict]


_sql_generator = None
_sql_executor = None
_summarizer = None
_cache = QueryCache()
_auditor = AuditLogger()


def _get_generator():
    global _sql_generator
    if _sql_generator is None:
        _sql_generator = SQLGeneratorAgent()
    return _sql_generator


def _get_executor():
    global _sql_executor
    if _sql_executor is None:
        _sql_executor = SQLExecutorAgent()
    return _sql_executor


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = SummarizerAgent()
    return _summarizer



def _accumulate_usage(state: PipelineState, new_usage: dict) -> dict:
    """Merge new token usage into the running total and compute cost."""
    prev = state.get("llm_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0}
    prompt = prev["prompt_tokens"] + new_usage.get("prompt_tokens", 0)
    completion = prev["completion_tokens"] + new_usage.get("completion_tokens", 0)
    cost = compute_llm_cost(MODEL_NAME, prompt, completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_cost_usd": cost}


def check_cache(state: PipelineState) -> dict:
    if not state.get("use_cache", True):
        return {"cache_hit": None}
    hit = _cache.get(state["question"])
    if hit:
        sql = hit.get("sql_query", "")
        df = None
        if sql:
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    result = conn.execute(sa_text(sql))
                    columns = list(result.keys())
                    rows = result.fetchall()
                df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame()
            except Exception:
                pass
        return {
            "sql_query": sql,
            "query_results": hit.get("query_results", ""),
            "summary": hit.get("summary", ""),
            "cache_hit": hit.get("cache_type", "exact"),
            "dataframe": df,
            "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0},
        }
    return {"cache_hit": None}


def schema_link(state: PipelineState) -> dict:
    engine = get_engine()
    schema_str, tables = _link_schema_fn(state["question"], engine)
    sample = get_sample_rows(engine)

    fk_graph = get_fk_graph(engine)
    fk_info = format_fk_info(fk_graph, tables)

    col_samples = get_column_samples(engine, tables)
    col_hints = format_column_hints(col_samples)

    return {
        "schema_info": schema_str,
        "sample_rows": sample,
        "linked_tables": tables,
        "fk_info": fk_info,
        "column_hints": col_hints,
    }


def _fix_dialect_sql(sql: str) -> str:
    """Auto-correct common cross-dialect mistakes in generated SQL."""
    if config.DB_DIALECT == "mysql":
        sql = re.sub(
            r"strftime\(\s*'%Y'\s*,\s*([^)]+)\)",
            r"YEAR(\1)",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r"strftime\(\s*'%m'\s*,\s*([^)]+)\)",
            r"MONTH(\1)",
            sql,
            flags=re.IGNORECASE,
        )
    return sql


def generate_sql(state: PipelineState) -> dict:
    error_context = state.get("error") or ""
    gen = _get_generator()
    sql = gen.generate(
        question=state["question"],
        schema_info=state["schema_info"],
        sample_rows=state.get("sample_rows", ""),
        error_context=error_context,
        fk_info=state.get("fk_info", ""),
        column_hints=state.get("column_hints", ""),
    )
    sql = _fix_dialect_sql(sql)
    usage = _accumulate_usage(state, gen.last_usage)
    return {"sql_query": sql, "error": None, "llm_usage": usage}


def validate_sql(state: PipelineState) -> dict:
    schema_meta = get_schema_meta()
    result = _validate_sql_fn(state["sql_query"], config.DB_DIALECT, schema_meta)
    if not result["valid"]:
        error_msg = "SQL validation failed: " + "; ".join(result["errors"])
        return {
            "validation_ok": False,
            "validation_errors": result["errors"],
            "error": error_msg,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    return {
        "validation_ok": True,
        "validation_errors": [],
    }


def explain_check(state: PipelineState) -> dict:
    result = _get_executor().explain(state["sql_query"])
    return {
        "explain_plan": result["plan_text"],
        "explain_warnings": result["warnings"],
    }


def execute_sql(state: PipelineState) -> dict:
    result = _get_executor().execute(state["sql_query"])
    if result["success"]:
        return {
            "query_results": result["data"],
            "dataframe": result["dataframe"],
            "error": None,
        }
    return {
        "query_results": "",
        "dataframe": None,
        "error": result["error"],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def summarize(state: PipelineState) -> dict:
    summ = _get_summarizer()
    summary = summ.summarize(
        question=state["question"],
        sql_query=state["sql_query"],
        query_results=state["query_results"],
    )
    usage = _accumulate_usage(state, summ.last_usage)
    return {"summary": summary, "llm_usage": usage}


def store_cache(state: PipelineState) -> dict:
    _cache.put(
        question=state["question"],
        sql_query=state["sql_query"],
        query_results=state["query_results"],
        summary=state["summary"],
    )
    return {}


def audit_log(state: PipelineState) -> dict:
    elapsed = None
    start = state.get("_start_time")
    if start:
        elapsed = time.time() - start

    _auditor.log({
        "question": state.get("question", ""),
        "linked_tables": state.get("linked_tables"),
        "sql_query": state.get("sql_query", ""),
        "validation_ok": state.get("validation_ok"),
        "validation_errors": state.get("validation_errors"),
        "explain_plan": state.get("explain_plan", ""),
        "explain_warnings": state.get("explain_warnings"),
        "execution_success": bool(state.get("query_results")),
        "query_results": state.get("query_results", ""),
        "summary": state.get("summary", ""),
        "cache_hit": state.get("cache_hit"),
        "elapsed_seconds": elapsed,
        "error": state.get("error"),
    })
    return {}


def format_error(state: PipelineState) -> dict:
    return {
        "summary": (
            f"Sorry, I couldn't execute the query after {MAX_RETRIES} retries.\n\n"
            f"**Last SQL attempted:**\n```sql\n{state.get('sql_query', 'N/A')}\n```\n\n"
            f"**Error:** {state.get('error', 'Unknown error')}"
        )
    }



def should_use_cache(state: PipelineState) -> str:
    if state.get("cache_hit"):
        return "cached"
    return "generate"


def should_retry_validation(state: PipelineState) -> str:
    if state.get("validation_ok"):
        return "valid"
    if state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    return "fail"


def should_retry_execution(state: PipelineState) -> str:
    if state.get("error") and state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    if state.get("error"):
        return "fail"
    return "continue"



def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("check_cache", check_cache)
    graph.add_node("schema_link", schema_link)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("explain_check", explain_check)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("summarize", summarize)
    graph.add_node("store_cache", store_cache)
    graph.add_node("format_error", format_error)
    graph.add_node("audit_log", audit_log)

    graph.set_entry_point("check_cache")

    # cache hit -> audit then done; miss -> schema link
    graph.add_conditional_edges(
        "check_cache",
        should_use_cache,
        {"cached": "audit_log", "generate": "schema_link"},
    )

    graph.add_edge("schema_link", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    # validation pass -> explain; fail -> retry or error
    graph.add_conditional_edges(
        "validate_sql",
        should_retry_validation,
        {"valid": "explain_check", "retry": "generate_sql", "fail": "format_error"},
    )

    graph.add_edge("explain_check", "execute_sql")

    # execution success -> summarize; fail -> retry or error
    graph.add_conditional_edges(
        "execute_sql",
        should_retry_execution,
        {"continue": "summarize", "retry": "generate_sql", "fail": "format_error"},
    )

    graph.add_edge("summarize", "store_cache")
    graph.add_edge("store_cache", "audit_log")
    graph.add_edge("format_error", "audit_log")
    graph.add_edge("audit_log", END)

    return graph.compile()


pipeline = build_graph()


def run_pipeline(question: str, use_cache: bool = True) -> PipelineState:
    """Run the full pipeline for a natural language question."""
    initial_state: PipelineState = {
        "question": question,
        "schema_info": "",
        "sample_rows": "",
        "sql_query": "",
        "query_results": "",
        "dataframe": None,
        "summary": "",
        "error": None,
        "retry_count": 0,
        "cache_hit": None,
        "use_cache": use_cache,
        "linked_tables": None,
        "validation_ok": None,
        "validation_errors": None,
        "explain_plan": None,
        "explain_warnings": None,
        "fk_info": None,
        "column_hints": None,
        "_start_time": time.time(),
        "llm_usage": None,
    }

    result = pipeline.invoke(initial_state)
    return result


def clear_cache():
    _cache.clear()


def reset_agents():
    """Clear cached agent singletons so they pick up a new DB engine/dialect."""
    global _sql_generator, _sql_executor, _summarizer
    _sql_generator = None
    _sql_executor = None
    _summarizer = None
    _reset_entity_vocab()
