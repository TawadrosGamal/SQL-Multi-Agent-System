from __future__ import annotations
import time
from config import (
    select_db_from_args,
    get_fk_graph,
    get_table_row_counts,
    get_column_samples,
    format_column_hints,
    format_fk_info,
)
from pipeline import (
    run_pipeline,
    clear_cache,
)
from validators.sql_validator import validate_sql
from validators.schema_catalog import SchemaCatalog
from audit.audit_logger import AuditLogger
from agents.sql_executor import (
    SQLExecutorAgent,
    _parse_explain_warnings,
    _extract_estimated_rows,
)
from cache.query_cache import (
    QueryCache,
    _keyword_overlap,
)
from validators.schema_linker import link_schema

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: list[tuple[str, str, bool, str]] = []


def record(phase: str, name: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        tag = "PASS"
    else:
        FAIL_COUNT += 1
        tag = "FAIL"
    RESULTS.append((phase, name, passed, detail))
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail and not passed else ""))


def run_test(phase: str, name: str, fn):
    try:
        passed, detail = fn()
        record(phase, name, passed, detail)
    except Exception as e:
        record(phase, name, False, f"Exception: {e}")


# =====================================================================
# PHASE 1: UNIT TESTS (no LLM calls)
# =====================================================================

def phase1_unit_tests():
    print("\n" + "=" * 70)
    print("PHASE 1: UNIT TESTS (no LLM calls)")
    print("=" * 70)

    # --- 1. Enriched Schema Metadata (config.py) ---
    print("\n--- 1. Enriched Schema Metadata ---")

    def test_fk_graph():
        
        fk = get_fk_graph()
        if not isinstance(fk, dict):
            return False, f"Expected dict, got {type(fk)}"
        if "orders" not in fk:
            return False, f"Missing 'orders' key; keys = {list(fk.keys())}"
        if not isinstance(fk["orders"], list):
            return False, f"Expected list for orders FKs, got {type(fk['orders'])}"
        return True, f"FK graph: {len(fk)} tables"

    def test_row_counts():
        counts = get_table_row_counts()
        if "orders" not in counts:
            return False, f"Missing 'orders'; keys = {list(counts.keys())}"
        if counts["orders"] < 1000:
            return False, f"orders has {counts['orders']} rows (expected ~9994)"
        return True, f"orders: {counts['orders']} rows"

    def test_column_samples():
        samples = get_column_samples(tables=["orders"])
        if not isinstance(samples, dict):
            return False, f"Expected dict, got {type(samples)}"
        expected_keys = {"orders.region", "orders.category", "orders.segment"}
        found = set(samples.keys())
        missing = expected_keys - found
        if missing:
            return False, f"Missing columns: {missing}; found: {found}"
        if "East" not in samples.get("orders.region", []):
            return False, f"Expected 'East' in region values"
        return True, f"{len(samples)} low-cardinality columns found"

    def test_format_column_hints():
        samples = get_column_samples(tables=["orders"])
        hints = format_column_hints(samples)
        if not hints.startswith("COLUMN VALUE HINTS:"):
            return False, f"Bad header: {hints[:50]}"
        if "orders.region" not in hints:
            return False, "Missing orders.region"
        return True, f"{len(hints)} chars"

    def test_format_fk_info():
        fk = get_fk_graph()
        info = format_fk_info(fk)
        if info and not info.startswith("FOREIGN KEY"):
            return False, f"Bad header: {info[:50]}"
        return True, "Empty (no FKs in single-table DB)" if not info else f"{len(info)} chars"

    run_test("Unit", "get_fk_graph()", test_fk_graph)
    run_test("Unit", "get_table_row_counts()", test_row_counts)
    run_test("Unit", "get_column_samples()", test_column_samples)
    run_test("Unit", "format_column_hints()", test_format_column_hints)
    run_test("Unit", "format_fk_info()", test_format_fk_info)

    # --- 2. SQL Validator ---
    print("\n--- 2. SQL Validator ---")

    def test_valid_select():
        schema = {"orders": ["order_id", "region", "category", "sale_price", "quantity"]}
        r = validate_sql("SELECT region, SUM(sale_price) FROM orders GROUP BY region", "sqlite", schema)
        if not r["valid"]:
            return False, f"Errors: {r['errors']}"
        return True, ""

    def test_forbidden_insert():

        schema = {"orders": ["order_id"]}
        r = validate_sql("INSERT INTO orders VALUES (1)", "sqlite", schema)
        if r["valid"]:
            return False, "INSERT should be rejected"
        if "forbidden" not in r["errors"][0].lower():
            return False, f"Wrong error: {r['errors']}"
        return True, ""

    def test_forbidden_delete():
        schema = {"orders": ["order_id"]}
        r = validate_sql("DELETE FROM orders WHERE order_id = 1", "sqlite", schema)
        if r["valid"]:
            return False, "DELETE should be rejected"
        return True, ""

    def test_unknown_table():
        schema = {"orders": ["order_id"]}
        r = validate_sql("SELECT * FROM nonexistent", "sqlite", schema)
        if r["valid"]:
            return False, "Unknown table should fail"
        if "unknown table" not in r["errors"][0].lower():
            return False, f"Wrong error: {r['errors']}"
        return True, ""

    def test_cte_not_flagged():
        schema = {"orders": ["order_id", "sale_price", "quantity", "product_id"]}
        sql = """
        WITH revenue AS (
            SELECT product_id, SUM(sale_price * quantity) AS total
            FROM orders GROUP BY product_id
        )
        SELECT * FROM revenue ORDER BY total DESC LIMIT 10
        """
        r = validate_sql(sql, "sqlite", schema)
        if not r["valid"]:
            return False, f"CTE 'revenue' flagged: {r['errors']}"
        return True, ""

    def test_syntax_error():
        schema = {"orders": ["order_id"]}
        r = validate_sql("SELEC * FORM orders", "sqlite", schema)
        if r["valid"]:
            return False, "Syntax error should be caught"
        return True, f"Error: {r['errors'][0][:60]}"

    run_test("Unit", "Valid SELECT passes", test_valid_select)
    run_test("Unit", "INSERT blocked", test_forbidden_insert)
    run_test("Unit", "DELETE blocked", test_forbidden_delete)
    run_test("Unit", "Unknown table caught", test_unknown_table)
    run_test("Unit", "CTE name not flagged", test_cte_not_flagged)
    run_test("Unit", "Syntax error caught", test_syntax_error)

    # --- 3. Schema Catalog ---
    print("\n--- 3. Schema Catalog ---")

    def test_catalog_build():
        cat = SchemaCatalog()
        cat.ensure_built()
        return True, "Built without error"

    def test_catalog_search():
        cat = SchemaCatalog()
        results = cat.search_tables("revenue from products", top_k=3)
        if not results:
            return False, "No results returned"
        if "orders" not in results:
            return False, f"Expected 'orders', got {results}"
        return True, f"Found: {results}"

    def test_catalog_descriptions():
        cat = SchemaCatalog()
        descs = cat.get_all_descriptions()
        if not descs:
            return False, "No descriptions"
        if "orders" not in descs:
            return False, f"Missing 'orders'; keys = {list(descs.keys())}"
        desc = descs["orders"]
        if "sale_price" not in desc:
            return False, f"Description missing 'sale_price': {desc[:80]}"
        return True, f"{len(descs)} table(s)"

    def test_catalog_single_desc():
        cat = SchemaCatalog()
        desc = cat.get_description("orders")
        if not desc:
            return False, "No description returned"
        if "order" not in desc.lower():
            return False, f"Bad description: {desc[:80]}"
        return True, f"Len: {len(desc)} chars"

    run_test("Unit", "Schema catalog builds", test_catalog_build)
    run_test("Unit", "Catalog search returns 'orders'", test_catalog_search)
    run_test("Unit", "Catalog get_all_descriptions()", test_catalog_descriptions)
    run_test("Unit", "Catalog get_description('orders')", test_catalog_single_desc)

    # --- 4. Cache Mechanics ---
    print("\n--- 4. Cache Mechanics ---")

    def test_exact_cache_roundtrip():
        c = QueryCache()
        c.put_exact("test_q_unit", "SELECT 1", "result_1", "summary_1")
        hit = c.get_exact("test_q_unit")
        if not hit:
            return False, "No hit after put"
        if hit["sql_query"] != "SELECT 1":
            return False, f"Wrong sql: {hit['sql_query']}"
        return True, ""

    def test_keyword_overlap_similar():
        score = _keyword_overlap(
            "Top 10 highest revenue products",
            "Top 10 products by highest revenue"
        )
        if score < 0.5:
            return False, f"Score too low for similar questions: {score:.2f}"
        return True, f"Overlap: {score:.2f}"

    def test_keyword_overlap_different():
        score = _keyword_overlap(
            "Top 10 highest revenue products",
            "What is the weather today in London"
        )
        if score > 0.2:
            return False, f"Score too high for different questions: {score:.2f}"
        return True, f"Overlap: {score:.2f}"

    run_test("Unit", "Exact cache put/get roundtrip", test_exact_cache_roundtrip)
    run_test("Unit", "Keyword overlap: similar questions", test_keyword_overlap_similar)
    run_test("Unit", "Keyword overlap: different questions", test_keyword_overlap_different)

    # --- 5. Audit Logger ---
    print("\n--- 5. Audit Logger ---")

    def test_audit_log_write_read():
        import tempfile, os
        tmp = os.path.join(tempfile.gettempdir(), "test_audit.db")
        logger = AuditLogger(db_path=tmp)
        logger.clear()
        logger.log({"question": "test_q", "sql_query": "SELECT 1", "validation_ok": True})
        entries = logger.recent(limit=5)
        if not entries:
            return False, "No entries after log"
        if entries[0]["question"] != "test_q":
            return False, f"Wrong question: {entries[0]['question']}"
        return True, ""

    def test_audit_log_clear():
        import tempfile, os
        tmp = os.path.join(tempfile.gettempdir(), "test_audit_clear.db")
        logger = AuditLogger(db_path=tmp)
        logger.log({"question": "q1"})
        logger.clear()
        entries = logger.recent()
        if entries:
            return False, f"Expected 0 entries, got {len(entries)}"
        return True, ""

    run_test("Unit", "Audit log write + read", test_audit_log_write_read)
    run_test("Unit", "Audit log clear", test_audit_log_clear)

    # --- 6. EXPLAIN Plan Parser ---
    print("\n--- 6. EXPLAIN Plan Parser ---")

    def test_explain_returns_dict():
        executor = SQLExecutorAgent()
        result = executor.explain("SELECT * FROM orders LIMIT 5")
        if not isinstance(result, dict):
            return False, f"Expected dict, got {type(result)}"
        if "plan_text" not in result or "warnings" not in result:
            return False, f"Missing keys: {list(result.keys())}"
        if not result["plan_text"]:
            return False, "Empty plan_text"
        return True, f"Plan: {len(result['plan_text'])} chars, {len(result['warnings'])} warnings"

    def test_explain_scan_warning():
        plan = "2 | 0 | 216 | SCAN orders"
        warnings = _parse_explain_warnings(plan, "sqlite")
        has_scan = any("full table scan" in w.lower() for w in warnings)
        if not has_scan:
            return False, f"Expected scan warning, got: {warnings}"
        return True, ""

    def test_explain_temp_btree_warning():
        plan = "0 | 0 | 0 | USE TEMP B-TREE FOR ORDER BY"
        warnings = _parse_explain_warnings(plan, "sqlite")
        has_btree = any("temporary b-tree" in w.lower() for w in warnings)
        if not has_btree:
            return False, f"Expected btree warning, got: {warnings}"
        return True, ""

    def test_extract_estimated_rows():
        plan = "0 | 0 | 216 | SCAN orders (~2000000 rows)"
        rows = _extract_estimated_rows(plan, "sqlite")
        if rows != 2000000:
            return False, f"Expected 2000000, got {rows}"
        return True, ""

    def test_row_count_guard():
        from agents.sql_executor import _parse_explain_warnings
        plan = "0 | 0 | 216 | SCAN orders (~2000000 rows)"
        warnings = _parse_explain_warnings(plan, "sqlite")
        has_guard = any("estimated to scan" in w.lower() for w in warnings)
        if not has_guard:
            return False, f"Expected row guard warning, got: {warnings}"
        return True, ""

    def test_mysql_explain_parsing():
        from agents.sql_executor import _parse_explain_warnings
        plan = "type: ALL | rows: 500000 | Using filesort | Using temporary"
        warnings = _parse_explain_warnings(plan, "mysql")
        expected = ["full table scan", "filesort", "temporary"]
        for kw in expected:
            if not any(kw.lower() in w.lower() for w in warnings):
                return False, f"Missing '{kw}' warning; got: {warnings}"
        return True, f"{len(warnings)} warnings"

    run_test("Unit", "explain() returns plan dict", test_explain_returns_dict)
    run_test("Unit", "EXPLAIN: SCAN TABLE warning", test_explain_scan_warning)
    run_test("Unit", "EXPLAIN: TEMP B-TREE warning", test_explain_temp_btree_warning)
    run_test("Unit", "EXPLAIN: _extract_estimated_rows()", test_extract_estimated_rows)
    run_test("Unit", "EXPLAIN: row count guard fires", test_row_count_guard)
    run_test("Unit", "EXPLAIN: MySQL warning parsing", test_mysql_explain_parsing)


# =====================================================================
# PHASE 2: INTEGRATION TESTS (requires LLM calls)
# =====================================================================

QUESTIONS = [
    "Top 10 Highest Revenue Generating Products",
    "Top 5 Highest Selling Products in Each Region",
    "Month-over-Month Growth Comparison for 2022 and 2023 Sales",
    "Highest Sales Month for Each Category",
]


def phase2_integration_tests():
    print("\n" + "=" * 70)
    print("PHASE 2: INTEGRATION TESTS (LLM pipeline, fresh queries)")
    print("=" * 70)

    print("  Clearing cache...")
    clear_cache()
    print("  Cache cleared.")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n--- Q{i}: {q} ---")
        start = time.time()
        result = run_pipeline(q, use_cache=False)
        elapsed = time.time() - start

        def make_test(r, field, check, msg):
            def fn():
                val = r.get(field)
                ok, detail = check(val)
                return ok, detail
            return fn

        run_test("Integration", f"Q{i}: cache_hit is None",
                 lambda r=result: (r.get("cache_hit") is None, f"Got: {r.get('cache_hit')}"))

        run_test("Integration", f"Q{i}: linked_tables populated",
                 lambda r=result: (bool(r.get("linked_tables")), f"Tables: {r.get('linked_tables')}"))

        run_test("Integration", f"Q{i}: validation_ok is True",
                 lambda r=result: (r.get("validation_ok") is True, f"Errors: {r.get('validation_errors')}"))

        run_test("Integration", f"Q{i}: SQL generated",
                 lambda r=result: (bool(r.get("sql_query")), ""))

        run_test("Integration", f"Q{i}: query_results non-empty",
                 lambda r=result: (bool(r.get("query_results")), ""))

        run_test("Integration", f"Q{i}: summary non-empty",
                 lambda r=result: (bool(r.get("summary")), ""))

        run_test("Integration", f"Q{i}: column_hints present",
                 lambda r=result: (bool(r.get("column_hints")), ""))

        run_test("Integration", f"Q{i}: explain_plan present",
                 lambda r=result: (bool(r.get("explain_plan")), ""))

        run_test("Integration", f"Q{i}: no error",
                 lambda r=result: (r.get("error") is None, f"Error: {r.get('error')}"))

        print(f"  Completed in {elapsed:.1f}s")


# =====================================================================
# PHASE 3: CACHE TESTS
# =====================================================================

def phase3_cache_tests():
    print("\n" + "=" * 70)
    print("PHASE 3: CACHE TESTS")
    print("=" * 70)

    from pipeline import run_pipeline

    # --- Exact cache hits ---
    print("\n--- Exact Cache Hits (same questions) ---")
    for i, q in enumerate(QUESTIONS, 1):
        result = run_pipeline(q, use_cache=True)
        run_test("Cache", f"Q{i} exact cache hit",
                 lambda r=result: (r.get("cache_hit") == "exact", f"Got: {r.get('cache_hit')}"))

    # --- Semantic cache hits (rephrased) ---
    print("\n--- Semantic Cache Hits (rephrased questions) ---")
    REPHRASED = [
        "Which products generate the most revenue? Show top 10",
        "Show me the top 5 best selling products for each region",
        "Compare monthly sales growth between 2022 and 2023",
        "For each product category, which month had the highest sales?",
    ]
    for i, q in enumerate(REPHRASED, 1):
        result = run_pipeline(q, use_cache=True)
        hit = result.get("cache_hit")
        run_test("Cache", f"Rephrased Q{i} semantic cache hit",
                 lambda r=result: (r.get("cache_hit") in ("exact", "semantic"),
                                   f"Got: {r.get('cache_hit')}"))


# =====================================================================
# PHASE 4: EDGE CASE TESTS
# =====================================================================

def phase4_edge_tests():
    print("\n" + "=" * 70)
    print("PHASE 4: EDGE CASES (safety, error handling)")
    print("=" * 70)

    # --- Safety ---
    print("\n--- Safety / Security ---")

    def test_executor_blocks_delete():
        from agents.sql_executor import SQLExecutorAgent
        executor = SQLExecutorAgent()
        result = executor.execute("DELETE FROM orders WHERE order_id = 1")
        if result["success"]:
            return False, "DELETE should be blocked"
        if "forbidden" not in result["error"].lower() and "blocked" not in result["error"].lower():
            return False, f"Wrong error: {result['error']}"
        return True, ""

    def test_executor_blocks_drop():
        from agents.sql_executor import SQLExecutorAgent
        executor = SQLExecutorAgent()
        result = executor.execute("DROP TABLE orders")
        if result["success"]:
            return False, "DROP should be blocked"
        return True, ""

    def test_executor_blocks_update():
        from agents.sql_executor import SQLExecutorAgent
        executor = SQLExecutorAgent()
        result = executor.execute("UPDATE orders SET region = 'X'")
        if result["success"]:
            return False, "UPDATE should be blocked"
        return True, ""

    run_test("Edge", "Executor blocks DELETE", test_executor_blocks_delete)
    run_test("Edge", "Executor blocks DROP", test_executor_blocks_drop)
    run_test("Edge", "Executor blocks UPDATE", test_executor_blocks_update)

    # --- Error handling ---
    print("\n--- Error Handling ---")

    def test_invalid_sql_execution():
        
        executor = SQLExecutorAgent()
        result = executor.execute("SELECT * FROM totally_fake_table_xyz")
        if result["success"]:
            return False, "Should fail for nonexistent table"
        if not result["error"]:
            return False, "Error message should be set"
        return True, f"Error: {result['error'][:60]}"

    def test_schema_linker_with_question():
        
        schema_str, tables = link_schema("total profit by category")
        if not tables:
            return False, "No tables returned"
        if "orders" not in tables:
            return False, f"Expected 'orders', got {tables}"
        return True, f"Tables: {tables}"

    run_test("Edge", "Executor handles invalid SQL gracefully", test_invalid_sql_execution)
    run_test("Edge", "Schema linker returns tables", test_schema_linker_with_question)

    # --- Audit log completeness ---
    print("\n--- Audit Log Completeness ---")

    def test_audit_log_has_entries():
        auditor = AuditLogger()
        entries = auditor.recent(limit=20)
        if len(entries) < 4:
            return False, f"Expected >= 4 entries (from pipeline runs), got {len(entries)}"
        return True, f"{len(entries)} entries found"

    def test_audit_log_has_fresh_and_cached():
        auditor = AuditLogger()
        entries = auditor.recent(limit=20)
        has_miss = any(e.get("cache_hit") is None or e.get("cache_hit") == "" for e in entries)
        has_hit = any(e.get("cache_hit") and e.get("cache_hit") != "" for e in entries)
        if not has_miss:
            return False, "No cache-miss entries found"
        if not has_hit:
            return False, "No cache-hit entries found"
        return True, "Both cache-miss and cache-hit entries present"

    run_test("Edge", "Audit log has entries from pipeline", test_audit_log_has_entries)
    run_test("Edge", "Audit log has both fresh and cached entries", test_audit_log_has_fresh_and_cached)


# =====================================================================
# SUMMARY
# =====================================================================

def print_summary():
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    phases = {}
    for phase, name, passed, detail in RESULTS:
        if phase not in phases:
            phases[phase] = {"pass": 0, "fail": 0, "tests": []}
        if passed:
            phases[phase]["pass"] += 1
        else:
            phases[phase]["fail"] += 1
        phases[phase]["tests"].append((name, passed, detail))

    for phase, data in phases.items():
        total = data["pass"] + data["fail"]
        print(f"\n  {phase}: {data['pass']}/{total} passed", end="")
        if data["fail"]:
            print(f"  ({data['fail']} FAILED)")
        else:
            print("  (all passed)")

        for name, passed, detail in data["tests"]:
            if not passed:
                print(f"    FAIL: {name}")
                if detail:
                    print(f"          {detail}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {PASS_COUNT + FAIL_COUNT}")
    if FAIL_COUNT == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{FAIL_COUNT} TEST(S) FAILED")
    print("=" * 70)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    select_db_from_args("Comprehensive test suite")

    phase1_unit_tests()
    phase2_integration_tests()
    phase3_cache_tests()
    phase4_edge_tests()
    print_summary()
