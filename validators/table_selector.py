from __future__ import annotations
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import require_openai_key, MODEL_NAME

_SYSTEM = (
    "You are a database schema expert. The user will give you a natural-language "
    "question and a list of database tables with short descriptions. Return ONLY "
    "a JSON array of the table names that are needed to answer the question. "
    "Do not explain. Example output: [\"orders\", \"products\"]"
)


def select_tables(
    question: str,
    table_descriptions: dict[str, str],
    max_tables: int = 5,
) -> list[str]:
    """Ask the LLM to pick the relevant tables from a name+description list.

    Parameters
    ----------
    question : str
        Natural-language user question.
    table_descriptions : dict[str, str]
        ``{table_name: one_line_description}``
    max_tables : int
        Hard cap on tables returned.

    Returns
    -------
    list[str]
        Table names chosen by the LLM (lower-cased).
    """
    table_list = "\n".join(f"- {name}: {desc}" for name, desc in table_descriptions.items())

    user_msg = (
        f"Question: {question}\n\n"
        f"Available tables:\n{table_list}\n\n"
        f"Return the JSON array of table names needed (max {max_tables})."
    )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=require_openai_key(),
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
    return _parse_table_list(response.content, table_descriptions, max_tables)


def _parse_table_list(
    raw: str,
    valid_tables: dict[str, str],
    max_tables: int,
) -> list[str]:
    """Parse the LLM's JSON array response, falling back to keyword extraction."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            names = [t.lower().strip() for t in parsed if isinstance(t, str)]
            names = [n for n in names if n in valid_tables]
            return names[:max_tables]
    except (json.JSONDecodeError, TypeError):
        pass

    valid_lower = {k.lower() for k in valid_tables}
    found = [w for w in raw.lower().split() if w.strip('[],"\'') in valid_lower]
    return list(dict.fromkeys(found))[:max_tables]
