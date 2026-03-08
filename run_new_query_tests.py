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


# =====================================================================
# DIVERSE QUESTIONS
# =====================================================================

QUESTIONS = [
    # Simple aggregations
    ("Q1", "What is the total profit for each region?"),
    ("Q2", "How many orders were placed in 2023?"),
    # Filtering + sorting
    ("Q3", "Show all orders in the Technology category with profit greater than 100, sorted by profit descending"),
    ("Q4", "Which ship mode is used most frequently?"),
    # Multi-column aggregation
    ("Q5", "What is the average discount and average sale price per segment?"),
    # Ranking / window
    ("Q6", "Which sub-category has the highest total revenue in each region?"),
    # Date-based
    ("Q7", "What are the total sales by quarter for 2022?"),
    ("Q8", "Which month in 2023 had the lowest total profit?"),
    # Conditional / complex
    ("Q9", "Compare the total number of orders and total revenue between the Consumer and Corporate segments"),
    ("Q10", "What percentage of total revenue comes from the West region?"),
]

# Rephrasings that should hit the semantic cache
REPHRASINGS = [
    ("R1 (rephrase Q1)", "Show me total profits broken down by region"),
    ("R2 (rephrase Q2)", "What is the total count of orders placed in the year 2023?"),
    ("R3 (rephrase Q8)", "In 2023, which month generated the least profit?"),
]


def validate_result(label: str, result: dict) -> None:
    """Run all checks on a pipeline result."""
    record(f"{label}: validation_ok",
           result.get("validation_ok") is True,
           f"Errors: {result.get('validation_errors')}")

    record(f"{label}: query_results non-empty",
           bool(result.get("query_results")),
           "")

    record(f"{label}: summary non-empty",
           bool(result.get("summary")),
           "")

    record(f"{label}: no error",
           result.get("error") is None,
           f"Error: {result.get('error')}")

    record(f"{label}: linked_tables populated",
           bool(result.get("linked_tables")),
           "")

    record(f"{label}: column_hints populated",
           bool(result.get("column_hints")),
           "")


def main():
    # ------------------------------------------------------------------
    # Phase 1: Fresh queries (no cache)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: 10 DIVERSE QUERIES (fresh, cache disabled)")
    print("=" * 70)

    print("\nClearing cache...")
    clear_cache()
    print("Cache cleared.\n")

    for label, question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"{label}: {question}")
        print("=" * 60)

        start = time.time()
        result = run_pipeline(question, use_cache=False)
        elapsed = time.time() - start
        TIMINGS.append((label, elapsed))

        # Print generated SQL
        sql = result.get("sql_query", "N/A")
        print(f"\n  SQL:\n    {sql.replace(chr(10), chr(10) + '    ')}")

        # Print first 5 lines of results
        raw = result.get("query_results", "")
        lines = raw.split("\n")
        preview = "\n    ".join(lines[:6])
        if len(lines) > 6:
            preview += f"\n    ... ({len(lines)} total lines)"
        print(f"\n  Results:\n    {preview}")

        # Print summary snippet
        summary = result.get("summary", "")
        snippet = summary[:200] + "..." if len(summary) > 200 else summary
        print(f"\n  Summary: {snippet}")

        print(f"\n  Time: {elapsed:.1f}s")

        # Validate
        validate_result(label, result)

    # ------------------------------------------------------------------
    # Phase 2: Semantic cache hits (rephrasings)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PHASE 2: SEMANTIC CACHE HITS (3 rephrasings)")
    print("=" * 70)

    # First re-run original questions WITH cache enabled so they get stored
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

        record(f"{label}: semantic cache hit",
               cache_hit in ("exact", "semantic"),
               f"Got: {cache_hit}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for _, p, _ in RESULTS if p)
    fail_count = sum(1 for _, p, _ in RESULTS if not p)

    print(f"\n  Fresh queries: {len(QUESTIONS)} questions, 6 checks each = {len(QUESTIONS) * 6} checks")
    print(f"  Cache tests:   {len(REPHRASINGS)} rephrasings")
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
        print(f"    None")

    print(f"\n{'='*70}")
    print(f"TOTAL: {pass_count} passed, {fail_count} failed out of {pass_count + fail_count}")
    if fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{fail_count} TEST(S) FAILED")
    print("=" * 70)


if __name__ == "__main__":

    select_db_from_args("Diverse query test suite")

    main()
