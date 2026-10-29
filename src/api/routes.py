from fastapi import APIRouter, HTTPException, Request
from openai import APIError

from src.api.schemas import SearchRequest, GraphSearchRequest, ChatRequest
from src.retrieval.query_normalizer import InvalidQueryError
from src.retrieval.legal_graph import GraphUnavailable
from src.retrieval.reranker import RerankerUnavailable
from src.rag.generator import GenerationError

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
def chat(payload: ChatRequest, request: Request):
    try:
        return request.app.state.chat.answer(payload.query, payload.top_k, include_evidence=payload.include_evidence,
                                             use_reranker=payload.use_reranker)
    except InvalidQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (GraphUnavailable, RerankerUnavailable, GenerationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Invalid retrieval or model output") from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail={"error": "OpenAI request failed",
                            "status_code": getattr(exc, "status_code", None), "code": getattr(exc, "code", None)}) from exc


@router.post("/search")
def search(payload: SearchRequest, request: Request):
    try:
        return request.app.state.retriever.retrieve(payload.query, payload.top_k)
    except InvalidQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (GraphUnavailable, RerankerUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="Embedding provider request failed") from exc


@router.post("/graph/search")
def graph_search(payload: GraphSearchRequest, request: Request):
    try:
        return request.app.state.retriever.graph_search(
            payload.doc_id, payload.article_no, max_hops=payload.max_hops, query=payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (GraphUnavailable, RerankerUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
