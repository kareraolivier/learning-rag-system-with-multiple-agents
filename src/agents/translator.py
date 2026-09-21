from langchain_core.messages import HumanMessage, SystemMessage

from src.models import get_agent_model
from src.state import RAGState


def translator_agent(state: RAGState):
    print("🇷🇼 [Agent 4] Translating execution payload to Kinyarwanda...")
    llm = get_agent_model("translator")
    sys_msg = SystemMessage(content="You are an expert translator. Translate the text into clear, modern Kinyarwanda. Output ONLY the clean translation.")
    response = llm.invoke([sys_msg, HumanMessage(content=state["english_response"])])
    print(f"🇷🇼 Kinyarwanda Output in translator: {response.content.strip()}")
    return {"final_output": response.content.strip()}
