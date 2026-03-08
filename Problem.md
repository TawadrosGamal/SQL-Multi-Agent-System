# Problem Statement: Natural Language to SQL for Business Intelligence

## The Challenge

In modern data-driven organizations, business users, analysts, and decision-makers constantly need to extract insights from relational databases. However, most of these users are not proficient in SQL, the primary language for querying structured data. This creates a critical bottleneck:

- **Data access is limited**: Non-technical stakeholders must rely on data engineers or analysts to write SQL queries for them, leading to delays and reduced productivity.
- **Knowledge gap**: Even when SQL is written, users may not understand the underlying schema—table names, column meanings, relationships—making it hard to formulate precise questions.
- **Error-prone manual process**: Hand-written SQL is susceptible to syntax errors, logical mistakes, and inefficient queries, especially when joins or aggregations are involved.
- **Lack of safety**: Ad-hoc queries might accidentally perform expensive full-table scans, time out, or even modify data if write permissions are misconfigured.
- **Context loss**: Results are often returned as raw tables, requiring further interpretation and summarization to be useful for business decisions.

## Why Existing Solutions Fall Short

Traditional natural language to SQL (NL2SQL) approaches, such as rule-based systems or early ML models, suffer from:

- **Poor generalization**: They cannot adapt to new database schemas or domain-specific terminology without extensive retraining.
- **Schema ignorance**: They often ignore foreign-key relationships, leading to incorrect joins and missing data.
- **No validation**: Generated SQL is not checked for syntactic correctness, semantic validity, or performance implications before execution.
- **No feedback loop**: Failed queries provide no actionable guidance, and there is no mechanism to learn from mistakes.
- **Security risks**: Direct execution of user-generated SQL can lead to data leaks or unintended modifications.

## The Need for an Intelligent, Multi-Agent Assistant

To bridge the gap between natural language questions and actionable data insights, a system must:

1. **Understand the user's intent** despite ambiguous phrasing or incomplete context.
2. **Map the question to the correct database entities**—tables, columns, and relationships—using both schema metadata and actual data samples.
3. **Generate syntactically correct SQL** tailored to the specific database dialect (SQLite, MySQL, etc.).
4. **Validate the SQL** for errors, potential performance issues, and safety (e.g., blocking writes, limiting rows).
5. **Execute the query safely** with timeouts and resource guards.
6. **Explain the results in natural language**, summarizing key findings for quick comprehension.
7. **Learn from past interactions** via caching and audit logging to improve future responses and enable traceability.
8. **Handle errors gracefully** with retries and fallback mechanisms to maximize success rate.

This project addresses these challenges by building a **multi-agent system powered by large language models (LLMs)** and orchestrated via LangGraph. It combines:

- **Schema linking** using embeddings and keyword matching to identify relevant tables and columns.
- **SQL generation** with dialect-specific prompts and foreign-key graph injection.
- **Robust validation** at multiple levels: syntax, semantics, and query cost.
- **Safe execution** with row limits and write-statement blocking.
- **Natural language summarization** of query results.
- **Two-level caching** (exact and semantic) to reduce latency and cost.
- **Comprehensive auditing** for monitoring and debugging.

The result is a tool that democratizes data access, enabling anyone to ask questions in plain English and receive instant, understandable answers—without writing a single line of SQL.