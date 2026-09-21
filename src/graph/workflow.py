from langgraph.graph import END, StateGraph

from src.agents.answer_generator import answer_generator_agent
from src.agents.query_generator import query_generator_agent
from src.agents.retriever import retriever_node
from src.state import RAGState


def build_agent_app():
    workflow = StateGraph(RAGState)
    workflow.add_node("agent_1", query_generator_agent)
    workflow.add_node("agent_2", retriever_node)
    workflow.add_node("agent_3", answer_generator_agent)
    # workflow.add_node("agent_4", translator_agent)

    workflow.set_entry_point("agent_1")
    workflow.add_edge("agent_1", "agent_2")
    workflow.add_edge("agent_2", "agent_3")
    # workflow.add_edge("agent_3", "agent_4")
    workflow.add_edge("agent_3", END)
    return workflow.compile()


agent_app = build_agent_app()
