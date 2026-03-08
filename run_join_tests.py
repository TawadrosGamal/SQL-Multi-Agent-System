from __future__ import annotations
import time
from config import select_db_from_args
from pipeline import run_pipeline, clear_cache

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: list[tuple[str, bool, str]] = []
TIMINGS: list[tuple[str, float]] = []


def record(name: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        tag = "PASS"
    else:
        FAIL_COUNT += 1
        tag = "FAIL"
    RESULTS.append((name, passed, detail))
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail and not passed else ""))


QUESTIONS = [
    # Simple JOINs
    (
        "J1",
        "List all orders with their product name and category by joining the orders and products tables. Show order_id, product_name, and category. Limit to 20 rows.",
    ),
    (
        "J2",
        "Show customer names and their total number of orders by joining customers and orders tables",
    ),
    # Aggregations with JOINs
    (
        "J3",
        "What is the total revenue per product category? Join orders with products on product_id",
    ),
    (
        "J4",
        "Which customer has the highest total profit? Join orders with customers on customer_id",
    ),
    # Multi-table JOINs (3 tables)
    (
        "J5",
        "Show customer_name, product_name, and order_date for orders in the West region. Join all three tables: orders, products, and customers. Limit to 15 rows.",
    ),
    (
        "J6",
        "What is the average sale_price per customer segment? Join orders with customers on customer_id",
    ),
    # Complex JOINs
    (
        "J7",
        "Which product sub_category generates the most revenue per customer segment? Join orders with both products and customers tables. Show segment, sub_category, and total revenue.",
    ),
    (
        "J8",
        "List the top 5 customers by total order quantity. Join orders with customers on customer_id. Show customer_name and total_quantity.",
    ),
    # Verification JOINs
    (
        "J9",
        "How many distinct products were ordered in each region? Join orders with products on product_id. Show region and the count of distinct product_ids.",
    ),
    (
        "J10",
        "Compare total revenue by product category between Consumer and Corporate segments. Join orders with products and customers tables.",
    ),
]

REPHRASINGS = [
    (
        "R1 (rephrase J3)",
        "Calculate the sum of revenue for each category of products by joining the products and orders tables",
    ),
    (
        "R2 (rephrase J2)",
        "Show the total number of orders for each customer name by joining customers and orders",
    ),
]


def validate_result(label: str, result: dict) -> None:
    """Run all checks on a pipeline result."""
    record(
        f"{label}: validation_ok",
        result.get("validation_ok") is True,
        f"Errors: {result.get('validation_errors')}",
    )

    record(
        f"{label}: query_results non-empty",
        bool(result.get("query_results")),
        "",
    )

    record(
        f"{label}: summary non-empty",
        bool(result.get("summary")),
        "",
    )

    record(
        f"{label}: no error",
        result.get("error") is None,
        f"Error: {result.get('error')}",
    )

    linked = result.get("linked_tables") or []
    record(
        f"{label}: linked_tables has multiple tables",
        len(linked) >= 2,
        f"Tables: {linked}",
    )

    record(
        f"{label}: fk_info populated",
        bool(result.get("fk_info")),
        "",
    )


def main():

    # ------------------------------------------------------------------
    # Phase 1: Fresh JOIN queries (no cache)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: 10 JOIN QUERIES (fresh, cache disabled)")
    print("=" * 70)

    print("\nClearing cache...")
    clear_cache()
    print("Cache cleared.\n")

    for label, question in QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"{label}: {question}")
        print("=" * 60)

        start = time.time()
        result = run_pipeline(question, use_cache=False)
        elapsed = time.time() - start
        TIMINGS.append((label, elapsed))

        sql = result.get("sql_query", "N/A")
        print(f"\n  SQL:\n    {sql.replace(chr(10), chr(10) + '    ')}")

        raw = result.get("query_results", "")
        lines = raw.split("\n")
        preview = "\n    ".join(lines[:6])
        if len(lines) > 6:
            preview += f"\n    ... ({len(lines)} total lines)"
        print(f"\n  Results:\n    {preview}")

        summary = result.get("summary", "")
        snippet = summary[:200] + "..." if len(summary) > 200 else summary
        print(f"\n  Summary: {snippet}")

        linked = result.get("linked_tables", [])
        fk = result.get("fk_info", "")
        print(f"\n  Linked tables: {linked}")
        print(f"  FK info: {fk[:120]}..." if len(fk) > 120 else f"  FK info: {fk}")
        print(f"  Time: {elapsed:.1f}s")

        validate_result(label, result)

    # ------------------------------------------------------------------
    # Phase 2: Semantic cache hits (rephrasings)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PHASE 2: SEMANTIC CACHE HITS (2 rephrasings)")
    print("=" * 70)

    print("\nStoring original questions in cache...")
    for label, question in QUESTIONS:
        run_pipeline(question, use_cache=True)
    print("Done.\n")

    for label, question in REPHRASINGS:
        print(f"\n--- {label}: {question} ---")
        start = time.time()
        result = run_pipeline(question, use_cache=True)
        elapsed = time.time() - start

        cache_hit = result.get("cache_hit")
        print(f"  Cache hit: {cache_hit}")
        print(f"  Time: {elapsed:.1f}s")

        record(
            f"{label}: semantic cache hit",
            cache_hit in ("exact", "semantic"),
            f"Got: {cache_hit}",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("JOIN TEST RESULTS SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for _, p, _ in RESULTS if p)
    fail_count = sum(1 for _, p, _ in RESULTS if not p)

    print(f"\n  JOIN queries: {len(QUESTIONS)} questions, 6 checks each = {len(QUESTIONS) * 6} checks")
    print(f"  Cache tests:  {len(REPHRASINGS)} rephrasings")
    print(f"\n  Timing breakdown:")
    for label, elapsed in TIMINGS:
        print(f"    {label}: {elapsed:.1f}s")
    total_time = sum(e for _, e in TIMINGS)
    print(f"    Total pipeline time: {total_time:.1f}s")

    print(f"\n  Failures:")
    failures = [(n, d) for n, p, d in RESULTS if not p]
    if failures:
        for name, detail in failures:
            print(f"    FAIL: {name}" + (f" -- {detail}" if detail else ""))
    else:
        print("    None")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {pass_count} passed, {fail_count} failed out of {pass_count + fail_count}")
    if fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{fail_count} TEST(S) FAILED")
    print("=" * 70)


if __name__ == "__main__":

    select_db_from_args("JOIN query test suite")

    main()
