from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME, require_openai_key

SYSTEM_PROMPT = """\
You are a data analyst assistant. You receive a user's question and the raw SQL query results.
Your job is to produce a clear, concise natural language summary.

RULES:
1. Directly answer the user's question using the data provided.
2. Format monetary values with $ and two decimal places (e.g., $1,234.56).
3. Format percentages with one decimal place (e.g., 12.3%).
4. If the data is a ranked list, present it as a numbered list.
5. If the data compares groups, highlight the key differences.
6. Keep the summary under 300 words unless the data warrants more.
7. Be factual — only state what the data shows. Do not invent numbers."""

HUMAN_PROMPT = """\
User question: {question}

SQL query used:
{sql_query}

Raw results:
{query_results}

Please provide a natural language summary of these results."""


class SummarizerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=require_openai_key(),
            temperature=0.3,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm
        self.last_usage: dict = {}

    def summarize(self, question: str, sql_query: str, query_results: str) -> str:
        ai_message = self.chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "query_results": query_results,
        })
        self.last_usage = (ai_message.response_metadata or {}).get("token_usage", {})
        return ai_message.content
