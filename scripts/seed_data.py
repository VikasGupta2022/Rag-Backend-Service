"""
Seed script to ingest sample knowledge base documents into the running RAG service.
Usage:
    python scripts/seed_data.py
"""

import sys
import httpx

API_BASE_URL = "http://localhost:8000/api/v1"

SAMPLE_DOCS = [
    {
        "title": "PostgreSQL pgvector Extension",
        "source": "https://github.com/pgvector/pgvector",
        "content": """
        pgvector is an open-source vector similarity search extension for PostgreSQL.
        It supports L2 distance, inner product, and cosine distance.
        It provides both HNSW (Hierarchical Navigable Small World) and IVFFlat index types.
        HNSW creates a multilayer graph and offers faster build times and better query performance than IVFFlat.
        For production, creating an HNSW index with vector_cosine_ops is standard for semantic search.
        """,
    },
    {
        "title": "Reciprocal Rank Fusion (RRF)",
        "source": "Information Retrieval Best Practices",
        "content": """
        Reciprocal Rank Fusion (RRF) is a method that combines multiple retrieval result sets.
        It calculates a score for each document as sum(1 / (k + rank)), where k is typically 60.
        RRF does not require score normalization across different retrieval algorithms,
        making it ideal for fusing dense embeddings and sparse BM25/full-text keyword search rankings.
        This hybrid retrieval significantly outperforms standalone dense or sparse search.
        """,
    },
    {
        "title": "Google Gemini 1.5 Flash Architecture",
        "source": "Google DeepMind Documentation",
        "content": """
        Gemini 1.5 Flash is a lightweight, high-speed, cost-efficient model optimized for low-latency tasks.
        It features a massive context window of up to 1 million tokens.
        It supports multimodal reasoning, structured JSON outputs, and native token-by-token streaming.
        Its paired embedding model, text-embedding-004, outputs 768-dimensional normalized vectors,
        achieving top benchmarks on the Massive Text Embedding Benchmark (MTEB).
        """,
    },
]


def seed():
    print(f"Connecting to {API_BASE_URL}...")
    with httpx.Client(timeout=30.0) as client:
        # Check liveness
        try:
            live_resp = client.get(f"{API_BASE_URL}/health/live")
            if live_resp.status_code != 200:
                print("Error: API is not alive. Please start the service first.")
                sys.exit(1)
        except Exception as e:
            print(f"Could not reach API: {e}")
            print("Make sure you run: docker-compose -f docker/docker-compose.yml up -d")
            sys.exit(1)

        print("API is healthy. Ingesting sample documents...")
        for doc in SAMPLE_DOCS:
            resp = client.post(f"{API_BASE_URL}/documents", json=doc)
            if resp.status_code == 201:
                res_data = resp.json()
                print(f"  [OK] Ingested '{doc['title']}' (ID: {res_data['id']}, Chunks: {res_data['chunks_count']})")
            else:
                print(f"  [FAIL] Ingesting '{doc['title']}': {resp.text}")

        print("\nSeeding complete! You can now run sample queries against /api/v1/query")


if __name__ == "__main__":
    seed()
