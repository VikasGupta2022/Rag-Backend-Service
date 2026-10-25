import time
import json
from typing import List, Optional
from collections.abc import AsyncGenerator
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.session import get_db
from src.services.embedding import get_embedding_service, BaseEmbeddingService
from src.services.llm import get_llm_service, BaseLLMService
from src.services.retrieval import HybridRetriever
from src.services.cache import get_cache_service, CacheService
from src.core.security import verify_api_key

router = APIRouter(prefix="/query", tags=["RAG Query Engine"])
settings = get_settings()


# --- Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or search query")
    use_cache: bool = Field(True, description="Whether to check and update Redis cache")


class Citation(BaseModel):
    document_title: str
    chunk_index: int
    content: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool
    latency_ms: float
    citations: List[Citation]


def _format_context(hits: List[dict]) -> str:
    """Formats retrieved chunks into clean LLM context."""
    context_parts = []
    for idx, hit in enumerate(hits, start=1):
        title = hit.get("document_title", "Unknown")
        content = hit.get("content", "").strip()
        context_parts.append(f"[Source {idx} - Document: {title}]\n{content}")
    return "\n\n".join(context_parts)


# --- Endpoints ---
@router.post("", response_model=QueryResponse)
async def execute_query(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: BaseEmbeddingService = Depends(get_embedding_service),
    llm: BaseLLMService = Depends(get_llm_service),
    cache: CacheService = Depends(get_cache_service),
    _: str = Depends(verify_api_key),
):
    """
    Executes standard hybrid RAG query:
    1. Checks Redis cache
    2. Runs hybrid search (dense vector + full-text) with RRF
    3. Generates LLM response using retrieved context
    4. Caches and returns answer with citations
    """
    start_time = time.perf_counter()

    # 1. Check Redis Cache
    if payload.use_cache:
        cached_result = await cache.get(payload.query)
        if cached_result:
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            return QueryResponse(
                query=payload.query,
                answer=cached_result["answer"],
                cached=True,
                latency_ms=latency,
                citations=[Citation(**c) for c in cached_result.get("citations", [])],
            )

    # 2. Hybrid Retrieval with RRF
    retriever = HybridRetriever(
        embedding_service=embedder,
        rrf_k=settings.RRF_K,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        final_top_k=settings.FINAL_TOP_K,
    )
    hits = await retriever.retrieve(session=db, query=payload.query)

    # 3. Format Context & Generate Answer
    context_text = _format_context(hits)
    answer = await llm.generate(question=payload.query, context=context_text)

    citations = [
        Citation(
            document_title=h.get("document_title", "Document"),
            chunk_index=h.get("chunk_index", 0),
            content=h.get("content", ""),
            score=h.get("rrf_score", 0.0),
        )
        for h in hits
    ]

    latency = round((time.perf_counter() - start_time) * 1000, 2)

    # 4. Save to Redis Cache
    if payload.use_cache:
        await cache.set(
            payload.query,
            {
                "answer": answer,
                "citations": [c.model_dump() for c in citations],
            },
        )

    return QueryResponse(
        query=payload.query,
        answer=answer,
        cached=False,
        latency_ms=latency,
        citations=citations,
    )


@router.get("", response_model=QueryResponse)
async def execute_query_get(
    q: str = Query(..., min_length=1, description="Your question or search query"),
    use_cache: bool = Query(True, description="Whether to check and update Redis cache"),
    db: AsyncSession = Depends(get_db),
    embedder: BaseEmbeddingService = Depends(get_embedding_service),
    llm: BaseLLMService = Depends(get_llm_service),
    cache: CacheService = Depends(get_cache_service),
    _: str = Depends(verify_api_key),
):
    """
    Convenience GET query endpoint:
    Ask questions directly via plain query parameter 'q' without writing any JSON.
    """
    payload = QueryRequest(query=q, use_cache=use_cache)
    return await execute_query(
        payload=payload,
        db=db,
        embedder=embedder,
        llm=llm,
        cache=cache,
        _=_,
    )


@router.get("/stream")
async def execute_query_stream(
    q: str = Query(..., min_length=1, description="Question or search query"),
    db: AsyncSession = Depends(get_db),
    embedder: BaseEmbeddingService = Depends(get_embedding_service),
    llm: BaseLLMService = Depends(get_llm_service),
    _: str = Depends(verify_api_key),
):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint.
    Streams token-by-token completion from the LLM, concluding with citations.
    """
    retriever = HybridRetriever(
        embedding_service=embedder,
        rrf_k=settings.RRF_K,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        final_top_k=settings.FINAL_TOP_K,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Retrieve relevant context
        hits = await retriever.retrieve(session=db, query=q)
        context_text = _format_context(hits)

        # 2. Stream tokens from LLM
        async for token in llm.generate_stream(question=q, context=context_text):
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

        # 3. Stream citation metadata at end of response
        citations = [
            {
                "document_title": h.get("document_title", "Document"),
                "chunk_index": h.get("chunk_index", 0),
                "content": h.get("content", ""),
                "score": h.get("rrf_score", 0.0),
            }
            for h in hits
        ]
        citation_payload = json.dumps({"event": "citations", "citations": citations})
        yield f"data: {citation_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
