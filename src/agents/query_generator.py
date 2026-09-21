import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.models import get_agent_model
from src.state import RAGState

_QUESTION_PREFIX = re.compile(
    r"^(please\s+)?(what|how|why|when|where|who|can|could|would|do|does|did|is|are|explain|describe|tell|summarize|understand)"
    r"(\s+\w+){0,6}\s+(about|on|regarding|for)?\s*",
    re.IGNORECASE,
)


def _clean_search_query(raw: str, fallback: str) -> str:
    query = raw.strip().strip('"').strip("'")
    query = query.splitlines()[0].strip() if query else ""
    if query.count(".") >= 3 and " " not in query:
        query = query.replace(".", " ")
    query = query.replace("?", " ").strip(" :-")
    query = _QUESTION_PREFIX.sub("", query).strip()
    query = re.sub(r"\s+", " ", query)
    return query or fallback


def query_generator_agent(state: RAGState):
    print("\n🤖 [Agent 1] Optimizing search query...")
    llm = get_agent_model("query_generator")
    sys_msg = SystemMessage(content=(
        "You are a search-query engineer for a research-paper vector database.\n"
        "Turn the user's question into a KEYWORD search query for Agent 2. Do not write a question.\n"
        "Rules:\n"
        "- No question marks and no phrases like what, how, explain, understand, tell me.\n"
        "- Keep paper titles, method names, and technical terms.\n"
        "- Expand likely acronyms.\n"
        "- Use 5 to 12 content keywords. Do not dump every related subfield.\n"
        "- Use normal spaces. Output ONLY the search query.\n\n"
        "Example\n"
        "User: What do you know about AI\n"
        "Query: artificial intelligence large language models machine learning"
    ))
    response = llm.invoke([sys_msg, HumanMessage(content=state["user_question"])])
    optimized = _clean_search_query(response.content, state["user_question"])
    print(f"🔍 Optimized Query in query builder: {optimized}")
    return {"optimized_query": optimized}
