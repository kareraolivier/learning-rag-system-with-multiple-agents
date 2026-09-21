from typing import TypedDict


class RAGState(TypedDict):
    user_question: str
    optimized_query: str
    retrieved_docs: str
    english_response: str
    final_output: str
