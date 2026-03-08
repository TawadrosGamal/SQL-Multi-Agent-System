# Multi-Agent SQL Assistant with LLMs

Converts natural language questions into SQL queries, executes them against a database, and returns natural language summaries. Built with LangChain, LangGraph, and OpenAI.

## Architecture

Three LangChain agents orchestrated via a LangGraph `StateGraph` pipeline, with a Streamlit web UI.

### Module Dependencies

```mermaid
flowchart TD
    app["app.py"] --> config["config.py"]
    app --> pipeline["pipeline.py"]

    pipeline --> config
    pipeline --> gen["agents/sql_generator.py"]
    pipeline --> exec["agents/sql_executor.py"]
    pipeline --> sum["agents/summarizer.py"]
    pipeline --> cache["cache/query_cache.py"]
    pipeline --> val["validators/sql_validator.py"]
    pipeline --> link["validators/schema_linker.py"]
    pipeline --> audit["audit/audit_logger.py"]

    gen --> config
    exec --> config
    sum --> config
    cache --> config
    val --> config
    link --> config
    link --> cat["validators/schema_catalog.py"]
    link --> sel["validators/table_selector.py"]
    cat --> config
    sel --> config

    setup_db["setup_db.py"] --> config
    setup_extra["setup_extra_tables.py"] --> config
    setup_extra --> cat
```

### Pipeline Flow

```mermaid
flowchart TD
    Start([Question]) --> CheckCache{check_cache}
    CheckCache -->|hit| AuditLog
    CheckCache -->|miss| SchemaLink

    SchemaLink["schema_link"] --> GenerateSQL
    GenerateSQL["generate_sql"] --> ValidateSQL
    ValidateSQL{"validate_sql"}
    ValidateSQL -->|valid| ExplainCheck
    ValidateSQL -->|"invalid + retries left"| GenerateSQL
    ValidateSQL -->|"invalid + no retries"| FormatError

    ExplainCheck{"explain_check"}
    ExplainCheck -->|ok| ExecuteSQL
    ExplainCheck -->|"costly + retries left"| GenerateSQL

    ExecuteSQL["execute_sql"]
    ExecuteSQL -->|success| Summarize
    ExecuteSQL -->|"error + retries left"| GenerateSQL
    ExecuteSQL -->|"error + no retries"| FormatError

    Summarize["summarize"] --> StoreCache["store_cache"]
    StoreCache --> AuditLog
    FormatError["format_error"] --> AuditLog
    AuditLog["audit_log"] --> Done([Result])
```

## Database Schema

Three normalized tables with foreign key relationships:

```mermaid
erDiagram
    customers ||--o{ orders : "customer_id"
    products ||--o{ orders : "product_id"
    customers {
        INTEGER customer_id PK
        TEXT customer_name
        TEXT email
        TEXT segment
        TEXT region
    }
    products {
        TEXT product_id PK
        TEXT product_name
        TEXT category
        TEXT sub_category
    }
    orders {
        INTEGER order_id
        INTEGER customer_id FK
        TEXT product_id FK
        TIMESTAMP order_date
        TEXT ship_mode
        TEXT segment
        TEXT country
        TEXT city
        TEXT state
        INTEGER postal_code
        TEXT region
        TEXT category
        TEXT sub_category
        REAL discount
        REAL sale_price
        REAL profit
        INTEGER quantity
    }
```

| Table | Rows | Description |
|-------|------|-------------|
| `orders` | 9,994 | Retail order records with sales, profit, and shipping data |
| `products` | 1,862 | Distinct products with category and sub-category |
| `customers` | 50 | Synthetic customers with segment and region |

## Features

- SQL generation with dialect-specific hints (SQLite / MySQL)
- Safe query execution with row limits, timeouts, and write-statement blocking
- LLM-powered natural language summaries
- Retry logic (up to 2 retries on validation or execution failure)
- Embedding-based schema linking (ChromaDB) with keyword boost and progressive LLM narrowing
- Foreign key graph injection for correct JOIN generation
- Column value sampling for accurate WHERE clauses
- sqlglot-based SQL validation with CTE/subquery awareness
- EXPLAIN plan analysis and row count guards
- Two-level caching: exact match (SHA-256 hash) + semantic similarity (embedding cosine + keyword overlap)
- Audit logging to SQLite
- Dynamic database selection (SQLite / MySQL presets from environment variables)

## Project Structure

```
Kuwait/
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── setup_db.py               # ETL: download CSV, transform, load to database
├── setup_extra_tables.py     # Create products + customers tables with FK constraints
├── config.py                 # Configuration, DB connection, schema introspection
├── pipeline.py               # LangGraph orchestrator (10-node pipeline)
├── app.py                    # Streamlit web app
├── test_questions.py         # CLI test runner for the 4 interview questions
├── run_tests.py              # Comprehensive test suite (77 tests)
├── run_new_query_tests.py    # Diverse query test suite (63 tests)
├── run_join_tests.py         # Multi-table JOIN test suite (62 tests)
├── agents/
│   ├── sql_generator.py      # Agent 1: NL -> SQL
│   ├── sql_executor.py       # Agent 2: SQL -> results
│   └── summarizer.py         # Agent 3: results -> NL summary
├── validators/
│   ├── sql_validator.py      # sqlglot-based SQL validation
│   ├── schema_linker.py      # Hybrid embedding + keyword schema linking
│   ├── schema_catalog.py     # ChromaDB-backed table description catalog
│   └── table_selector.py     # Lightweight LLM table selector
├── cache/
│   ├── query_cache.py        # Two-level cache (exact + semantic)
│   └── chroma_store/         # ChromaDB persistent storage
└── audit/
    └── audit_logger.py       # SQLite-backed structured audit logging
```

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key with credits

### Installation

```bash
cd Kuwait
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `DB_SQLITE` | `sqlite:///retail_orders.db` | SQLite database preset |
| `DB_MYSQL` | *(optional)* | MySQL database preset (e.g. `mysql+pymysql://root:pass@localhost:3306/retail_orders`) |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model to use |

Any `DB_*=<connection-url>` environment variable is automatically registered as a selectable preset in the Streamlit sidebar.

### Database Setup

```bash
# Step 1: Load base orders data (~9,994 rows)
python setup_db.py

# Step 2: Create normalized tables (products, customers) with FK constraints
python setup_extra_tables.py
```

## Usage

### Streamlit Web App

```bash
streamlit run app.py
```

### CLI Test Runner

```bash
# 4 interview questions
python test_questions.py

# 77-test comprehensive suite
python run_tests.py

# 63-test diverse query suite
python run_new_query_tests.py

# 62-test multi-table JOIN suite
python run_join_tests.py
```

All test scripts accept `--db sqlite` or `--db mysql` to select the target database.
