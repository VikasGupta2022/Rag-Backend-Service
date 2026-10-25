import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.services.embedding import BaseEmbeddingService
from src.services.in_memory_store import get_in_memory_store

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridRetriever:
    """
    Executes dense vector search and sparse full-text search,
    merging candidates via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedding_service: BaseEmbeddingService,
        rrf_k: int = 60,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        final_top_k: int = 5,
    ):
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k

    async def _dense_search(
        self, session: AsyncSession, query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """Queries pgvector for cosine similarity."""
        query_sql = text("""
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.metadata_json,
                d.title as document_title,
                1 - (c.embedding <=> cast(:query_vec as vector)) as similarity
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.embedding <=> cast(:query_vec as vector) ASC
            LIMIT :limit;
        """)

        str_vec = f"[{','.join(str(x) for x in query_vector)}]"
        result = await session.execute(
            query_sql, {"query_vec": str_vec, "limit": self.dense_top_k}
        )
        rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def _sparse_search(
        self, session: AsyncSession, query_text: str
    ) -> List[Dict[str, Any]]:
        """Queries PostgreSQL Full-Text Search via tsvector and tsquery."""
        query_sql = text("""
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.metadata_json,
                d.title as document_title,
                ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', :query)) as rank_score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
            ORDER BY rank_score DESC
            LIMIT :limit;
        """)

        result = await session.execute(
            query_sql, {"query": query_text, "limit": self.sparse_top_k}
        )
        rows = result.mappings().all()
        return [dict(row) for row in rows]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combines rankings from dense and sparse results using RRF:
        RRF_score(d) = sum(1 / (k + rank))
        """
        fused_scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = str(item["chunk_id"])
            item_map[chunk_id] = item
            score = 1.0 / (self.rrf_k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score

        # 2. Process sparse rankings
        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = str(item["chunk_id"])
            item_map[chunk_id] = item
            score = 1.0 / (self.rrf_k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score

        # 3. Sort by aggregated RRF score
        sorted_ids = sorted(
            fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True
        )

        results = []
        for cid in sorted_ids[: self.final_top_k]:
            doc = item_map[cid]
            doc["rrf_score"] = round(fused_scores[cid], 5)
            results.append(doc)

        return results

    async def retrieve(
        self, session: Optional[AsyncSession], query: str
    ) -> List[Dict[str, Any]]:
        """Main retrieval entry point: embeds query, runs hybrid search, and applies RRF."""
        # 1. Generate query embedding
        query_vector = await self.embedding_service.embed_query(query)

        # 2. Check if DB session is connected
        from src.db.session import is_db_connected

        if not is_db_connected or session is None:
            # In-memory hybrid retrieval
            store = get_in_memory_store()
            dense_hits = store.search_dense(query_vector, top_k=self.dense_top_k)
            sparse_hits = store.search_sparse(query, top_k=self.sparse_top_k)
            return self._reciprocal_rank_fusion(dense_hits, sparse_hits)

        # 3. PostgreSQL + pgvector hybrid searches
        try:
            dense_hits = await self._dense_search(session, query_vector)
            sparse_hits = await self._sparse_search(session, query)
            return self._reciprocal_rank_fusion(dense_hits, sparse_hits)
        except Exception as e:
            logger.warning(f"Database retrieval failed: {e}. Falling back to in-memory store.")
            store = get_in_memory_store()
            dense_hits = store.search_dense(query_vector, top_k=self.dense_top_k)
            sparse_hits = store.search_sparse(query, top_k=self.sparse_top_k)
            return self._reciprocal_rank_fusion(dense_hits, sparse_hits)
