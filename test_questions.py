from pipeline import run_pipeline
from config import select_db_from_args
QUESTIONS = [
    "Top 10 Highest Revenue Generating Products",
    "Top 5 Highest Selling Products in Each Region",
    "Month-over-Month Growth Comparison for 2022 and 2023 Sales",
    "Highest Sales Month for Each Category",
]


def main():
    select_db_from_args("Run 4 interview questions")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"Q{i}: {question}")
        print("=" * 80)

        result = run_pipeline(question)

        print(f"\n--- Cache Hit ---")
        print(result.get("cache_hit", "None (fresh query)"))

        print(f"\n--- Schema Linking ---")
        tables = result.get("linked_tables")
        print(f"Tables: {', '.join(tables)}" if tables else "N/A (cache hit)")

        fk = result.get("fk_info", "")
        if fk:
            print(f"FK Info:\n{fk}")

        hints = result.get("column_hints", "")
        if hints:
            print(f"Column Hints:\n{hints}")

        print(f"\n--- Generated SQL ---")
        print(result.get("sql_query", "N/A"))

        print(f"\n--- SQL Validation ---")
        if result.get("validation_ok") is True:
            print("PASSED")
        elif result.get("validation_ok") is False:
            print(f"FAILED: {result.get('validation_errors')}")
        else:
            print("N/A")

        print(f"\n--- EXPLAIN Plan ---")
        plan = result.get("explain_plan", "")
        if plan:
            print(plan[:500])
        else:
            print("N/A")
        warnings = result.get("explain_warnings") or []
        if warnings:
            print(f"Warnings: {warnings}")

        print(f"\n--- Raw Results ---")
        print(result.get("query_results", "N/A")[:2000])

        print(f"\n--- Summary ---")
        print(result.get("summary", "N/A"))

        if result.get("error"):
            print(f"\n--- Error ---")
            print(result["error"])


if __name__ == "__main__":
    main()
