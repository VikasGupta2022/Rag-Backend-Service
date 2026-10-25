from src.services.retrieval import HybridRetriever
from src.services.embedding import MockEmbeddingService


def test_reciprocal_rank_fusion_math():
    embedder = MockEmbeddingService()
    retriever = HybridRetriever(
        embedding_service=embedder,
        rrf_k=60,
        final_top_k=5,
    )

    # Chunk A appears rank 1 in dense, rank 2 in sparse
    # Chunk B appears rank 2 in dense, absent in sparse
    # Chunk C appears rank 1 in sparse, absent in dense
    dense_results = [
        {"chunk_id": "chunk-A", "content": "Text A", "document_title": "Doc 1"},
        {"chunk_id": "chunk-B", "content": "Text B", "document_title": "Doc 2"},
    ]
    sparse_results = [
        {"chunk_id": "chunk-C", "content": "Text C", "document_title": "Doc 3"},
        {"chunk_id": "chunk-A", "content": "Text A", "document_title": "Doc 1"},
    ]

    fused = retriever._reciprocal_rank_fusion(dense_results, sparse_results)

    # Chunk A: rank 1 dense (1/61) + rank 2 sparse (1/62) = ~0.01639 + 0.01613 = 0.03252
    # Chunk C: rank 1 sparse (1/61) = ~0.01639
    # Chunk B: rank 2 dense (1/62) = ~0.01613
    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "chunk-A"  # Highest combined score
    assert fused[1]["chunk_id"] == "chunk-C"
    assert fused[2]["chunk_id"] == "chunk-B"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
    assert fused[1]["rrf_score"] > fused[2]["rrf_score"]
