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


# Seed questions: run fresh to populate the cache
SEED_QUESTIONS = [
    ("S1", "Who is the best customer"),
    ("S2", "Give me total sales from order number 1 to order number 30"),
    ("S3", "Top 10 highest revenue generating products"),
    ("S4", "Total revenue by region in 2022"),
]

# Questions that SHOULD hit the semantic cache (synonyms / rephrasings)
SHOULD_HIT = [
    ("SYN1 (rephrase S1)", "Who is the top customer"),
    ("SYN2 (rephrase S1)", "Who is the best performing customer"),
    ("SYN3 (rephrase S3)", "Which products generate the most revenue? Show top 10"),
]

# Questions that must NOT hit the cache (different intent)
MUST_MISS = [
    ("MISS1 ordinal diff", "Who is the third best customer",
     "ordinal '3' absent from seed S1"),
    ("MISS2 year diff", "Total revenue by region in 2024",
     "year '2024' differs from seed S4 '2022'"),
    ("MISS3 entity diff", "Give me total sales from customer number 1 to customer number 30",
     "'customer' vs 'order' in seed S2"),
    ("MISS4 entity diff", "Give me total sales from product number 1 to product number 30",
     "'product' vs 'order' in seed S2"),
]


def validate_result(label: str, result: dict) -> None:
    record(
        f"{label}: validation_ok",
        result.get("validation_ok") is True,
        f"Errors: {result.get('validation_errors')}",
    )
    record(
        f"{label}: no error",
        result.get("error") is None,
        f"Error: {result.get('error')}",
    )
    record(
        f"{label}: summary non-empty",
        bool(result.get("summary")),
        "",
    )


def main():
    # ------------------------------------------------------------------
    # Phase 1: Seed the cache with fresh queries
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: SEED QUERIES (fresh, populating cache)")
    print("=" * 70)

    print("\nClearing cache...")
    clear_cache()
    print("Cache cleared.\n")

    for label, question in SEED_QUESTIONS:
        print(f"\n--- {label}: {question} ---")
        start = time.time()
        result = run_pipeline(question, use_cache=True)
        elapsed = time.time() - start
        TIMINGS.append((label, elapsed))

        sql = result.get("sql_query", "N/A")
        print(f"  SQL: {sql[:120]}")
        print(f"  Time: {elapsed:.1f}s")

        validate_result(label, result)

    # ------------------------------------------------------------------
    # Phase 2: Synonym equivalence (SHOULD hit cache)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PHASE 2: SYNONYM EQUIVALENCE (should hit semantic cache)")
    print("=" * 70)

    for label, question in SHOULD_HIT:
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
    # Phase 3: Differentiation (must NOT hit cache)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PHASE 3: DIFFERENTIATION (must NOT match cached questions)")
    print("=" * 70)

    for label, question, reason in MUST_MISS:
        print(f"\n--- {label}: {question} ---")
        print(f"  Reason: {reason}")
        start = time.time()
        result = run_pipeline(question, use_cache=True)
        elapsed = time.time() - start

        cache_hit = result.get("cache_hit")
        print(f"  Cache hit: {cache_hit}")
        print(f"  Time: {elapsed:.1f}s")

        record(
            f"{label}: cache miss (correct)",
            cache_hit is None,
            f"Got: {cache_hit} (should be None)",
        )

    # ------------------------------------------------------------------
    # Phase 4: Year validation (non-existent year should not crash)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("PHASE 4: YEAR VALIDATION (2025 not in dataset)")
    print("=" * 70)

    print("\n--- YV1: Total revenue by region in 2025 ---")
    start = time.time()
    result = run_pipeline("Total revenue by region in 2025", use_cache=False)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")

    record(
        "YV1: no crash on non-existent year",
        result.get("error") is None,
        f"Error: {result.get('error')}",
    )
    record(
        "YV1: summary produced",
        bool(result.get("summary")),
        "",
    )
    raw = result.get("query_results", "")
    record(
        "YV1: SQL executed (may return empty)",
        result.get("validation_ok") is True,
        f"Validation errors: {result.get('validation_errors')}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("EDGE-CASE TEST RESULTS SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for _, p, _ in RESULTS if p)
    fail_count = sum(1 for _, p, _ in RESULTS if not p)

    print(f"\n  Seed queries:    {len(SEED_QUESTIONS)} questions")
    print(f"  Synonym hits:    {len(SHOULD_HIT)} tests")
    print(f"  Must-miss:       {len(MUST_MISS)} tests")
    print(f"  Year validation: 1 test")

    print(f"\n  Timing breakdown:")
    for label, elapsed in TIMINGS:
        print(f"    {label}: {elapsed:.1f}s")

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

    select_db_from_args("Edge-case test suite")

    main()
