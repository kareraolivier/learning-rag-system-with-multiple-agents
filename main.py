import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.database import ingest_documents, DB_PATH
from src.graph import agent_app

# Initialize FastAPI App
app = FastAPI(
    title="Kinyarwanda Agentic RAG API", 
    description="Multi-agent pipeline running Qwen2.5 local nodes with LangGraph"
)

# Define request structure
class QueryRequest(BaseModel):
    question: str

# Define response structure
class QueryResponse(BaseModel):
    user_question: str
    optimized_query: str
    english_response: str
    # kinyarwanda_output: str

@app.on_event("startup")
def startup_event():
    """Ensures vector database exists before serving web endpoints."""
    if not os.path.exists(DB_PATH):
        print("⚠️ Chroma DB not found at startup. Running automatic ingestion...")
        ingest_documents()

@app.post("/api/query", response_model=QueryResponse)
async def process_query(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # Run input through the 4-agent LangGraph compilation
        initial_state = {"user_question": payload.question}
        result = await agent_app.ainvoke(initial_state) # Using async version
        
        return {
            "user_question": result.get("user_question", ""),
            "optimized_query": result.get("optimized_query", ""),
            "english_response": result.get("english_response", ""),
            # "kinyarwanda_output": result.get("final_output", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")

# CLI utility option to trigger standalone indexing
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        print("🚀 Starting independent ingestion pipeline...")
        ingest_documents()
    else:
        # Run via internal uvicorn call if script executed directly
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
