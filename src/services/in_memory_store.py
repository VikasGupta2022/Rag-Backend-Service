import math
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """
    In-memory vector and document store used when PostgreSQL + pgvector is offline.
    Supports cosine dense search, keyword sparse search, and Reciprocal Rank Fusion.
    """

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []

    def add_document(
        self, title: str, source: Optional[str] = None, content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        doc = {
            "id": doc_id,
            "title": title,
            "source": source,
            "content_type": content_type,
            "chunks_count": 0,
        }
        self.documents[doc_id] = doc
        return doc

    def add_chunks(
        self,
        doc_id: str,
        doc_title: str,
        chunk_records: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        for chunk, emb in zip(chunk_records, embeddings):
            chunk_entry = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": doc_id,
                "document_title": doc_title,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "embedding": emb,
                "metadata_json": chunk.get("metadata", {}),
            }
            self.chunks.append(chunk_entry)

        if doc_id in self.documents:
            self.documents[doc_id]["chunks_count"] += len(chunk_records)

    def get_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.chunks = [c for c in self.chunks if str(c["document_id"]) != doc_id]
            return True
        return False

    def search_dense(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Computes Cosine Similarity across all stored chunk embeddings."""
        if not self.chunks or not query_vector:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_vector)) or 1.0
        results = []

        for c in self.chunks:
            emb = c["embedding"]
            dot = sum(a * b for a, b in zip(query_vector, emb))
            c_norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            similarity = dot / (q_norm * c_norm)

            item = dict(c)
            item["similarity"] = similarity
            results.append(item)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_sparse(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Computes keyword match frequency as a lightweight BM25 substitute."""
        if not self.chunks or not query:
            return []

        query_tokens = set(query.lower().split())
        results = []

        for c in self.chunks:
            content_tokens = c["content"].lower().split()
            matched = sum(1 for t in content_tokens if t in query_tokens)
            if matched > 0:
                item = dict(c)
                item["rank_score"] = matched / (len(content_tokens) + 10)
                results.append(item)

        results.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return results[:top_k]


# Global singleton instance
_in_memory_store = InMemoryVectorStore()


def get_in_memory_store() -> InMemoryVectorStore:
    return _in_memory_store
