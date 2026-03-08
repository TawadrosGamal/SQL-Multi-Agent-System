import io
import json
import time
from typing import Any
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sa_text
from audit.audit_logger import AuditLogger
import config
from config import (
    get_db_presets,
    set_database,
    get_engine,
    get_schema_info,
    get_fk_graph,
    get_schema_meta,
    get_table_row_counts,
)
from pipeline import reset_agents, run_pipeline,clear_cache


st.set_page_config(
    page_title="Multi-Agent SQL Assistant",
    page_icon="🗄️",
    layout="wide",
)

SAMPLE_QUESTIONS = [
    "Top 10 Highest Revenue Generating Products",
    "Top 5 Highest Selling Products in Each Region",
    "Month-over-Month Growth Comparison for 2022 and 2023 Sales",
    "Highest Sales Month for Each Category",
    "Show me the top 10 products by total profits in 2023",
    "Compare total sales revenues for Q1 vs Q2 2023 by region",
]


def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []


def _test_connection(url: str) -> tuple[bool, str]:
    """Try a lightweight SELECT 1 and return (ok, message)."""
    try:
        eng = create_engine(url)
        with eng.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)[:120]


def main():
    init_session()

    st.title("Multi-Agent SQL Assistant")
    st.caption(
        "Natural language -> SQL -> Results -> Summary  |  "
        "Powered by LangChain + LangGraph + OpenAI"
    )

    with st.sidebar:

        presets = get_db_presets()
        preset_labels = list(presets.keys())

        if "selected_db" not in st.session_state:
            st.session_state.selected_db = preset_labels[0]

        st.header("Database")
        selected = st.selectbox(
            "Active database",
            preset_labels,
            index=preset_labels.index(st.session_state.selected_db)
            if st.session_state.selected_db in preset_labels
            else 0,
            key="db_selector",
        )

        if selected != st.session_state.selected_db:
            set_database(presets[selected])
            reset_agents()
            st.session_state.selected_db = selected
            st.toast(f"Switched to {selected}")
            st.rerun()

        ok, conn_msg = _test_connection(config.DATABASE_URL)
        if ok:
            st.success(f"Connected -- {config.DB_DIALECT.upper()}", icon="✅")
        else:
            st.error(f"Connection failed: {conn_msg}", icon="❌")

        if ok:
            try:
                engine = get_engine()
                meta = get_schema_meta(engine)
                row_counts = get_table_row_counts(engine)
                total_rows = sum(v for v in row_counts.values() if v > 0)

                st.markdown(
                    f"**Dialect:** `{config.DB_DIALECT}`  \n"
                    f"**Tables:** {len(meta)}  \n"
                    f"**Total rows:** {total_rows:,}"
                )
            except Exception as exc:
                st.warning(f"Could not load schema info: {exc}")

        st.divider()

        st.header("Sample Questions")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state["question_input"] = q

        st.divider()

        st.header("Settings")
        use_cache = st.toggle(
            "Enable cache", value=True, help="Skip LLM calls for repeated questions"
        )

        if st.button("Clear cache", use_container_width=True):
            clear_cache()
            st.toast("Cache cleared!")

        st.divider()

        with st.expander("Database Schema", expanded=False):
            try:
                engine = get_engine()
                schema_meta = get_schema_meta(engine)
                fk_graph = get_fk_graph(engine)

                for table, columns in sorted(schema_meta.items()):
                    fk_cols = {
                        e["column"]
                        for t, edges in fk_graph.items()
                        if t == table
                        for e in edges
                    }
                    col_data = []
                    for col in columns:
                        is_pk = col.endswith("_id") and col not in fk_cols and col == columns[0]
                        role = "PK" if is_pk else ("FK" if col in fk_cols else "")
                        col_data.append({"Column": col, "Key": role})
                    st.caption(f"**{table}** ({len(columns)} cols)")
                    st.dataframe(
                        pd.DataFrame(col_data),
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(col_data) * 35 + 38, 200),
                    )

                with st.container():
                    st.caption("Raw DDL")
                    schema_text = get_schema_info(engine)
                    st.code(schema_text, language="text")
            except Exception as e:
                st.error(str(e))

        st.subheader("Audit Log")
        _render_audit_log()

        st.divider()

        st.markdown(
            "**Agent 1** -- SQL Generator (LLM)  \n"
            "**Agent 2** -- SQL Executor (DB)  \n"
            "**Agent 3** -- Answer Summarizer (LLM)  \n"
            "---\n"
            "**Guards:** Schema Linker - SQL Validator - EXPLAIN Check - Audit Log"
        )

    question = st.text_input(
        "Ask a question about the retail orders data:",
        key="question_input",
        placeholder="e.g., What are the top 10 products by revenue?",
    )

    col_run, col_info = st.columns([1, 5])
    with col_run:
        run_btn = st.button("Run", type="primary", use_container_width=True)

    if run_btn and question:
        run_pipeline_ui(question, use_cache)
    elif run_btn and not question:
        st.warning("Please enter a question first.")

    if st.session_state.history:
        st.divider()
        st.subheader("Query History")
        for i, entry in enumerate(reversed(st.session_state.history)):
            label = (
                f"{'[CACHED] ' if entry.get('cache_hit') else ''}"
                f"Q: {entry['question']}"
            )
            with st.expander(label, expanded=(i == 0)):
                render_result(entry, idx=i)


def run_pipeline_ui(question: str, use_cache: bool):
    with st.status("Running pipeline...", expanded=True) as status:
        st.write(
            "Checking cache..." if use_cache else "Cache disabled -- generating fresh."
        )
        start = time.time()
        result = run_pipeline(question, use_cache=use_cache)
        elapsed = time.time() - start

        cache_hit = result.get("cache_hit")
        llm_usage = result.get("llm_usage") or {}
        cost_usd = llm_usage.get("total_cost_usd", 0.0)
        if cache_hit:
            status.update(
                label=f"Cache hit ({cache_hit}) -- {elapsed:.2f}s -- $0.00", state="complete"
            )
        else:
            status.update(label=f"Done in {elapsed:.1f}s -- ${cost_usd:.4f}", state="complete")

    entry = {
        "question": question,
        "sql_query": result.get("sql_query", ""),
        "query_results": result.get("query_results", ""),
        "dataframe": result.get("dataframe"),
        "summary": result.get("summary", ""),
        "error": result.get("error"),
        "cache_hit": cache_hit,
        "elapsed": elapsed,
        "linked_tables": result.get("linked_tables"),
        "validation_ok": result.get("validation_ok"),
        "validation_errors": result.get("validation_errors"),
        "explain_plan": result.get("explain_plan"),
        "explain_warnings": result.get("explain_warnings"),
        "fk_info": result.get("fk_info"),
        "column_hints": result.get("column_hints"),
        "llm_usage": llm_usage,
    }
    st.session_state.history.append(entry)
    render_result(entry, idx=-1)


_STAGES = [
    ("Cache", "cache_hit"),
    ("Schema Link", "linked_tables"),
    ("Validation", "validation_ok"),
    ("EXPLAIN", "explain_plan"),
    ("Execution", "query_results"),
    ("Summary", "summary"),
]


def _render_pipeline_stepper(entry: dict):
    """Render a horizontal stepper showing which pipeline stages completed."""
    cols = st.columns(len(_STAGES))

    cache_hit = entry.get("cache_hit")

    for col, (label, key) in zip(cols, _STAGES):
        with col:
            if label == "Cache":
                if cache_hit:
                    st.markdown(
                        f"**{label}**  \n"
                        f"🟢 Hit ({cache_hit})"
                    )
                else:
                    st.markdown(f"**{label}**  \n🔵 Miss")
            elif cache_hit and label != "Cache":
                st.markdown(f"**{label}**  \n⚪ Skipped")
            else:
                value = entry.get(key)
                if label == "Validation":
                    if value is True:
                        st.markdown(f"**{label}**  \n🟢 Passed")
                    elif value is False:
                        st.markdown(f"**{label}**  \n🔴 Failed")
                    else:
                        st.markdown(f"**{label}**  \n⚪ N/A")
                elif label == "EXPLAIN":
                    warns = entry.get("explain_warnings") or []
                    if warns:
                        st.markdown(
                            f"**{label}**  \n🟡 {len(warns)} warning(s)"
                        )
                    elif value:
                        st.markdown(f"**{label}**  \n🟢 Clean")
                    else:
                        st.markdown(f"**{label}**  \n⚪ N/A")
                elif label == "Execution":
                    if entry.get("error") and not value:
                        st.markdown(f"**{label}**  \n🔴 Error")
                    elif value:
                        st.markdown(f"**{label}**  \n🟢 OK")
                    else:
                        st.markdown(f"**{label}**  \n⚪ N/A")
                else:
                    if value:
                        st.markdown(f"**{label}**  \n🟢 Done")
                    else:
                        st.markdown(f"**{label}**  \n⚪ N/A")


def render_result(entry: dict, idx: int = 0):
    if entry.get("cache_hit"):
        st.info(f"Served from **{entry['cache_hit']}** cache")

    _render_pipeline_stepper(entry)

    llm_usage = entry.get("llm_usage") or {}
    cost_usd = llm_usage.get("total_cost_usd", 0.0)
    prompt_tok = llm_usage.get("prompt_tokens", 0)
    comp_tok = llm_usage.get("completion_tokens", 0)
    total_tok = prompt_tok + comp_tok

    cost_cols = st.columns([1, 1, 1, 3])
    with cost_cols[0]:
        st.metric("LLM Cost", f"${cost_usd:.4f}")
    with cost_cols[1]:
        st.metric("Prompt Tokens", f"{prompt_tok:,}")
    with cost_cols[2]:
        st.metric("Completion Tokens", f"{comp_tok:,}")

    st.markdown("### Summary")
    st.markdown(entry["summary"])

    col_sql, col_data = st.columns(2)

    with col_sql:
        st.markdown("**Generated SQL**")
        sql = entry.get("sql_query", "")
        if sql:
            st.code(sql, language="sql")
        else:
            st.caption("No SQL generated")

    with col_data:
        st.markdown("**Raw Results**")
        df = entry.get("dataframe")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.text(entry.get("query_results", "No results"))

    st.markdown("**Pipeline Details**")
    with st.container():
        detail_cols = st.columns(3)

        with detail_cols[0]:
            st.markdown("**Schema Linking**")
            tables = entry.get("linked_tables")
            if tables:
                st.write(f"Tables selected: {', '.join(tables)}")
            else:
                st.write("N/A (cache hit or not yet run)")
            fk = entry.get("fk_info")
            if fk:
                st.text(fk)
            hints = entry.get("column_hints")
            if hints:
                st.text(hints)

        with detail_cols[1]:
            st.markdown("**SQL Validation**")
            if entry.get("validation_ok") is True:
                st.success("Passed")
            elif entry.get("validation_ok") is False:
                errs = entry.get("validation_errors") or []
                st.error("Failed: " + "; ".join(errs))
            else:
                st.write("N/A")

        with detail_cols[2]:
            st.markdown("**EXPLAIN Plan**")
            warnings = entry.get("explain_warnings") or []
            if warnings:
                for w in warnings:
                    st.warning(w)
            elif entry.get("explain_plan"):
                st.success("No performance warnings")
            else:
                st.write("N/A")

        plan = entry.get("explain_plan")
        if plan:
            st.code(plan, language="text")

    # Export button
    df = entry.get("dataframe")
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download results as CSV",
            csv_buffer.getvalue(),
            file_name="query_results.csv",
            mime="text/csv",
            key=f"download_csv_{idx}",
        )

    if entry.get("error"):
        st.error(f"Error encountered: {entry['error']}")

    if entry.get("elapsed") is not None:
        cost_str = f" -- ${cost_usd:.4f}" if total_tok > 0 else " -- $0.00 (cached)"
        st.caption(f"Completed in {entry['elapsed']:.2f}s{cost_str}")


def _render_audit_log():
    """Show the most recent audit log entries in the sidebar."""
    try:
        auditor = AuditLogger()
        entries = auditor.recent(limit=20)
    except Exception as exc:
        st.error(f"Could not load audit log: {exc}")
        return

    if not entries:
        st.write("No audit entries yet.")
        return

    for entry in entries:
        q = entry.get("question", "?")
        cache = entry.get("cache_hit") or "miss"
        elapsed = entry.get("elapsed_seconds")
        elapsed_str = f"{elapsed:.2f}s" if elapsed else "?"
        validation = "pass" if entry.get("validation_ok") else "fail"
        warns = entry.get("explain_warnings", "")
        has_warns = bool(warns and warns != "null" and warns != "[]")

        label = f"[{cache}] {q[:50]}"
        with st.expander(label):
            st.write(f"**Elapsed:** {elapsed_str}")
            st.write(f"**Validation:** {validation}")
            if has_warns:
                try:
                    warn_list = (
                        json.loads(warns) if isinstance(warns, str) else warns
                    )
                    for w in warn_list:
                        st.warning(w)
                except (json.JSONDecodeError, TypeError):
                    st.write(warns)
            sql = entry.get("sql_query", "")
            if sql:
                st.code(sql, language="sql")


if __name__ == "__main__":
    main()
