from src.database import search_chunks
from src.state import RAGState


def retriever_node(state: RAGState):
    print("📂 [Agent 2] Extracting semantic contexts from Chroma DB...")
    search_query = (state.get("optimized_query") or "").strip() or state["user_question"]
    print(f"📂 Searching Chroma with: {search_query}")
    docs = search_chunks(search_query, k=4)
    if not docs and search_query != state["user_question"]:
        print("📂 No hits for optimized query, retrying with the original question...")
        docs = search_chunks(state["user_question"], k=4)

    context_text = "\n\n".join(
        f"{doc.page_content.strip()}"
        for doc in docs
        if doc.page_content.strip()
    )
    print(f"📂 Retrieved {len(docs)} docs ({len(context_text)} chars)")
    for i, doc in enumerate(docs, 1):
        preview = " ".join(doc.page_content.split())[:180]
        source = doc.metadata.get("source", "unknown")
        print(f"   {i}. {source}: {preview}")
    return {"retrieved_docs": context_text if context_text.strip() else "No relevant documents found."}
