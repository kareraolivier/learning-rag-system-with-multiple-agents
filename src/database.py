import os
import re

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_PATH = "./chroma_db"
DATA_DIR = "./data"
COLLECTION_NAME = "rag_papers"


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def _sanitize_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")


def ingest_documents():
    """Loads PDFs from ./data, embeds them, and writes vectors into chroma_db."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created '{DATA_DIR}' folder. Please place your PDFs there.")
        return None

    print("📄 Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print("⚠️ No PDFs found in the data directory. Ingestion skipped.")
        return None

    print(f"✂️ Splitting {len(docs)} document pages into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    for doc in splits:
        doc.page_content = _sanitize_text(doc.page_content)
        doc.metadata = {
            key: _sanitize_text(value) if isinstance(value, str) else value
            for key, value in doc.metadata.items()
        }
    splits = [doc for doc in splits if doc.page_content.strip()]

    print(f"💾 Saving {len(splits)} chunks to local Chroma DB at '{DB_PATH}'...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Ingestion complete! Stored {vectorstore._collection.count()} chunks.")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _open_vectorstore():
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
        create_collection_if_not_exists=False,
    )
    count = vectorstore._collection.count()
    if count == 0:
        raise RuntimeError(
            "Chroma collection is empty. Run `python main.py --ingest` after adding PDFs to ./data."
        )
    print(f"📂 Opened Chroma collection '{COLLECTION_NAME}' with {count} chunks.")
    return vectorstore


def _looks_like_references(text: str, page: int | None = None) -> bool:
    """Bibliography chunks match lots of keywords but have almost no usable claims."""
    years = len(re.findall(r"\b(19|20)\d{2}[a-z]?\b", text))
    venues = len(re.findall(
        r"\b(inproceedings|arxiv preprint|international conference on learning)\b",
        text,
        re.I,
    ))
    # Early pages often cite a few papers; do not treat that as a bibliography.
    if page is not None and page <= 2:
        return venues >= 2 or years >= 8
    if years >= 3:
        return True
    if venues >= 2:
        return True
    stripped = text.strip()
    if stripped[:1].islower() and years >= 1 and venues >= 1:
        return True
    return False


def search_chunks(query: str, k: int = 4):
    """Prefer early-page paper body over late-page bibliographies."""
    vectorstore = _open_vectorstore()
    early = vectorstore.similarity_search(query, k=max(10, k * 3), filter={"page": {"$lte": 2}})
    chosen = []
    seen = set()
    for doc in early:
        page = doc.metadata.get("page")
        if _looks_like_references(doc.page_content, page if isinstance(page, int) else None):
            continue
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(doc)
        if len(chosen) >= k:
            return chosen

    extra = vectorstore.similarity_search(query, k=40)
    for doc in extra:
        page = doc.metadata.get("page")
        page_num = page if isinstance(page, int) else None
        if page_num is not None and page_num > 4:
            continue
        if _looks_like_references(doc.page_content, page_num):
            continue
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(doc)
        if len(chosen) >= k:
            break
    return chosen[:k] or extra[:k]


def get_retriever():
    """Opens the existing paper collection. Does not create an empty one."""
    return _open_vectorstore().as_retriever(search_kwargs={"k": 4})
