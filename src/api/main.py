import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.routes import router
from src.embedding.vector_store import ChromaVectorStore
from src.retrieval.retriever import LegalRetriever
from src.retrieval.config import DEFAULT_RETRIEVAL_CONFIG
from src.rag.generator import AnswerGenerator, LLMConfig, DEFAULT_LLM_CONFIG
from src.rag.service import LegalChatService

ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(ROOT / ".env")
    config = Path(os.getenv("EMBEDDING_CONFIG", "src/embedding/config.yaml"))
    if not config.is_absolute():
        config = ROOT / config
    if not config.is_file():
        raise FileNotFoundError(f"Embedding config does not exist: {config}")
    app.state.store = ChromaVectorStore.from_config(config, ROOT)
    retrieval_path = Path(os.getenv("RETRIEVAL_CONFIG", str(DEFAULT_RETRIEVAL_CONFIG)))
    if not retrieval_path.is_absolute():
        retrieval_path = ROOT / retrieval_path
    try:
        app.state.retriever = LegalRetriever.from_config(app.state.store, retrieval_path, ROOT)
        llm_path = Path(os.getenv("LLM_CONFIG", str(DEFAULT_LLM_CONFIG)))
        if not llm_path.is_absolute():
            llm_path = ROOT / llm_path
        app.state.chat = LegalChatService(app.state.retriever, AnswerGenerator(LLMConfig.from_yaml(llm_path)))
        yield
    finally:
        app.state.store.close()


app = FastAPI(title="Vietnam Legal RAG", lifespan=lifespan)
app.include_router(router)
