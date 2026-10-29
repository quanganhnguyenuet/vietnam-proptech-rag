from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=100)


class ChatRequest(SearchRequest):
    include_evidence: bool = True
    use_reranker: bool | None = None


class GraphSearchRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_]+$", max_length=100)
    article_no: str = Field(pattern=r"^\d+[a-zđ]?$", max_length=12)
    max_hops: int = Field(default=2, ge=1, le=3)
    query: str | None = Field(default=None, min_length=1, max_length=1000)
