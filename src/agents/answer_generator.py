from langchain_core.messages import HumanMessage, SystemMessage

from src.models import  
from src.state import RAGState


def _analysis_focus(state: RAGState) -> str:
    optimized = (state.get("optimized_query") or "").strip()
    return optimized or state["user_question"]


def answer_generator_agent(state: RAGState):
    print("🧠 [Agent 3] Generating response from fetched facts...")
    llm = get_agent_model("answer_generator")
    question = (state.get("user_question") or "").strip() or _analysis_focus(state)
    print(f"🧠 Agent 3 answering: {question}")
    sys_msg = SystemMessage(content=(
        "You are a helpful assistant answering from a local research-paper library.\n"
        "Write the reply a person would actually want to read.\n"
        "Rules:\n"
        "- Answer the user's question directly in 1-3 short paragraphs.\n"
        "- Never mention excerpts, retrieved documents, chunks, or 'these papers discuss'.\n"
        "- Do not list paper titles, benchmarks, ISO standards, or citations unless asked.\n"
        "- If the sources are thin or off-topic, say this library only has limited detail, then share only what is clearly supported.\n"
        "- Do not invent titles, numbers, or findings.\n"
        "- Speak to the user in plain English. Lead with the answer.\n\n"
        "Bad answer: These papers discuss various aspects of AI, including large language models...\n"
        "Good answer: This library is mostly recent AI research, especially large language models. "
        "Some work speeds up big models with mixture-of-experts while keeping accuracy; "
        "other work looks at how AI should be run in production systems."
    ))
    prompt = (
        f"User question: {question}\n\n"
        f"Source excerpts:\n{state['retrieved_docs']}\n\n"
        "Answer the user question. These excerpts come from recent papers, not a general encyclopedia."
    )
    response = llm.invoke([sys_msg, HumanMessage(content=prompt)])
    print(f"🧠 English Response in answer generator: {response.content.strip()}")
    return {"english_response": response.content.strip()}
