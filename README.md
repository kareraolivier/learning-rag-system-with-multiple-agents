# Multi-agent RAG API

Local retrieval-augmented generation over your own PDFs. A FastAPI service runs a LangGraph pipeline: one agent rewrites the question into a search query, another retrieves chunks from Chroma, and a third answers from those excerpts. Chat models are configured per agent and created through a provider factory (Ollama by default).

## GitHub repository extras

**Description** (About blurb):

> Local multi-agent RAG over PDFs: LangGraph + Chroma + FastAPI, with swappable chat models per agent (Ollama by default).

**Topics:** `rag` `langchain` `langgraph` `ollama` `chromadb` `fastapi` `multi-agent` `python`

## How it works

```
POST /api/query { "question": "..." }
        │
        ▼
  Agent 1  query generator   → keyword search query
        │
        ▼
  Agent 2  retriever         → top chunks from chroma_db
        │
        ▼
  Agent 3  answer generator  → English answer from those chunks
```

A Kinyarwanda translator agent exists in `src/agents/translator.py` but is not wired into the graph yet.

Qwen never reads the PDFs directly. It only sees what Chroma returns. Embeddings use `nomic-embed-text` via Ollama.

## Prerequisites

- Python 3.11+ (3.14 works with the current venv)
- [Ollama](https://ollama.com) running locally
- These Ollama models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put PDFs in `./data`, then build the vector index:

```bash
python main.py --ingest
```

Start the API:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Query the API

```bash
curl -sS -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What do you know about Dual-Interest Sequential Product Recommendation?"}'
```

Response shape:

```json
{
  "user_question": "...",
  "optimized_query": "...",
  "english_response": "..."
}
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Configure models

Defaults are `ollama` / `qwen2.5:3b`. Override in the environment (this project does not auto-load `.env`; export the vars or use your shell).

| Role | Variables |
|---|---|
| All agents (fallback) | `DEFAULT_MODEL_PROVIDER`, `DEFAULT_MODEL`, `DEFAULT_MODEL_TEMPERATURE` |
| Query generator | `QUERY_GENERATOR_MODEL`, `QUERY_GENERATOR_MODEL_PROVIDER` |
| Answer generator | `ANSWER_GENERATOR_MODEL`, `ANSWER_GENERATOR_MODEL_PROVIDER` |
| Translator | `TRANSLATOR_MODEL`, `TRANSLATOR_MODEL_PROVIDER` |

Example: keep the rewriter local and answer with OpenAI:

```bash
export QUERY_GENERATOR_MODEL=qwen2.5:3b
export ANSWER_GENERATOR_MODEL_PROVIDER=openai
export ANSWER_GENERATOR_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...
```

Install the matching LangChain package for any non-Ollama provider (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, …). Agents do not import a specific vendor; `src/models/factory.py` calls `init_chat_model`.

## Project layout

```
main.py                 FastAPI app and --ingest CLI
src/state.py            shared LangGraph state
src/database.py         PDF ingest + Chroma search
src/models/             provider factory and env settings
src/agents/             one file per agent
src/graph/              LangGraph workflow only
data/                   PDFs (not committed)
chroma_db/              vector index (not committed)
```

## Notes

- Re-run `python main.py --ingest` after you add or replace PDFs.
- Broad questions such as “What is AI?” are answered from this paper library, not from a general encyclopedia.
- `.env` and `data/*.pdf` are gitignored. Keep `.env.example` as the template.
