from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
from config import (
    MODEL_NAME,
    require_openai_key,
)   


_DIALECT_HINTS = {
    "mysql": (
        "Use MySQL syntax (LIMIT not TOP, backticks for identifiers if needed).\n"
        "CRITICAL: Use YEAR(column) for year extraction and MONTH(column) for month extraction. "
        "NEVER use strftime() — it does not exist in MySQL.\n"
        "Use ROUND(value, 2) for rounding.\n"
        "IMPORTANT: Do NOT use MySQL reserved words (RANK, ROW, ROWS, GROUPS, etc.) as column aliases. "
        "Use descriptive alternatives like `sales_rank`, `row_num`, etc."
    ),
    "sqlite": (
        "Use SQLite syntax (LIMIT, double-quotes or no quotes for identifiers).\n"
        "Use strftime('%Y', order_date) for year and strftime('%m', order_date) for month.\n"
        "CAST to REAL for decimal division. Use ROUND(value, 2) for rounding."
    ),
}

SYSTEM_PROMPT = """\
You are an expert SQL query generator for a {dialect} database.

DATABASE SCHEMA:
{schema_info}

{fk_info}

{column_hints}

SAMPLE DATA:
{sample_rows}

RULES:
1. Output ONLY the raw SQL query. No markdown fences, no explanations, no comments.
2. {dialect_hints}
3. Always alias aggregated columns with meaningful names.
4. Use CTEs (WITH clauses) for complex queries involving rankings or comparisons.
5. For "top N per group" queries, use ROW_NUMBER() window function.
6. For month-over-month comparisons, pivot with CASE WHEN on the year.
7. Revenue = SUM(sale_price * quantity). Profit = SUM(profit). Sales/Selling = SUM(quantity).
8. Round monetary values to 2 decimal places.
9. Only generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, or ALTER.
10. Use the FOREIGN KEY RELATIONSHIPS above to correctly JOIN related tables.
11. Use the COLUMN VALUE HINTS above to write accurate WHERE clauses with valid values. Only the values listed in the hints exist in the data.
12. When the question mentions a specific year, month, or date range, ALWAYS include a WHERE clause filtering on the date column. Never omit temporal filters.

{error_context}"""

HUMAN_PROMPT = "Question: {question}"


class SQLGeneratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=require_openai_key(),
            temperature=0,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm
        self.last_usage: dict = {}

    def generate(
        self,
        question: str,
        schema_info: str,
        sample_rows: str = "",
        error_context: str = "",
        fk_info: str = "",
        column_hints: str = "",
    ) -> str:
        if error_context:
            error_context = (
                f"\nPREVIOUS ATTEMPT FAILED. Fix the SQL based on this error:\n{error_context}"
            )

        dialect = config.DB_DIALECT
        dialect_hints = _DIALECT_HINTS.get(dialect, _DIALECT_HINTS["sqlite"])

        ai_message = self.chain.invoke({
            "question": question,
            "schema_info": schema_info,
            "sample_rows": sample_rows,
            "error_context": error_context,
            "dialect": dialect,
            "dialect_hints": dialect_hints,
            "fk_info": fk_info or "",
            "column_hints": column_hints or "",
        })

        self.last_usage = (ai_message.response_metadata or {}).get("token_usage", {})

        sql = ai_message.content.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return sql.strip()
