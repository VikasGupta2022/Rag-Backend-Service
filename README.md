# 🚀 High-Performance RAG & AI Backend Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Google Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D.svg?logo=redis&logoColor=white)](https://redis.io)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A production-grade, enterprise-ready **Retrieval-Augmented Generation (RAG)** backend service engineered in Python. It features **hybrid search** (dense embeddings + sparse keyword search with Reciprocal Rank Fusion), **Redis query caching**, real-time **Server-Sent Events (SSE) token streaming**, and full distributed tracing.

---

## 🏛 Architecture

```mermaid
flowchart TD
    Client["Client / Frontend"] -->|HTTP / SSE Stream| API["FastAPI Application"]
    
    subgraph "Query & Retrieval Pipeline"
        API -->|1. Check Cache| RedisCache[("Redis Query Cache")]
        RedisCache -.->|Hit (<15ms)| API
        RedisCache -->|Miss: Query| QueryEngine["Hybrid Search Engine"]
        
        QueryEngine -->|Dense Vector Cosine| PGVector[("PostgreSQL pgvector (HNSW)")]
        QueryEngine -->|Sparse Full-Text| PGText[("PostgreSQL GIN (tsvector)")]
        
        PGVector --> RRF["Reciprocal Rank Fusion (RRF)"]
        PGText --> RRF
        
        RRF --> ContextBuilder["Context Assembler"]
        ContextBuilder --> Gemini["Google Gemini 1.5 Flash"]
        Gemini -->|Token-by-Token Stream| API
    end

    subgraph "Ingestion Pipeline"
        DocUpload["Document / File Upload"] --> API
        API --> Chunker["Semantic Text Chunker"]
        Chunker --> Embedder["Gemini text-embedding-004 (768d)"]
        Embedder -->|Batch Upsert| PGVector
    end
```

---

## ✨ Key Features

* **⚡ Hybrid Search with RRF (Reciprocal Rank Fusion)**: Combines dense vector cosine similarity with PostgreSQL native full-text search (`tsvector` / `tsquery`). RRF normalizes scores using $RRF(d) = \sum \frac{1}{60 + \text{rank}}$ to dramatically improve retrieval accuracy.
* **🌐 Google Gemini 1.5 Flash + text-embedding-004**: Generates 768-dimensional embeddings and streams responses token-by-token with low latency.
* **🏎 Redis Query Caching**: Stores previous queries and responses, slashing repeat request latency from ~1.5s down to **< 15ms**.
* **📡 Real-Time Server-Sent Events (SSE)**: Dedicated `/api/v1/query/stream` endpoint emitting live tokens as they are generated, concluding with full citation metadata.
* **🔍 Observability & Tracing**: Custom middleware injects distributed `X-Correlation-ID` tracing and `X-Process-Time-Ms` duration headers on all HTTP responses.
* **🛡 Resilient Architecture**: Automatic fallback to mock embeddings and LLM generation for offline testing or when `GEMINI_API_KEY` is not yet configured.

---

## 📁 Repository Structure

```
rag-backend-service/
├── .github/workflows/
│   └── ci.yml               # Automated Ruff linting, formatting & Pytest CI
├── docker/
│   ├── Dockerfile           # Optimized multi-stage container
│   └── docker-compose.yml   # App + pgvector + Redis
├── scripts/
│   └── seed_data.py         # Knowledge base seeder script
├── src/
│   ├── api/
│   │   ├── middleware.py    # Correlation ID & execution timing
│   │   └── v1/
│   │       ├── health.py    # Liveness and readiness health checks
│   │       ├── documents.py # Document ingestion & chunk management
│   │       ├── query.py     # Hybrid search & SSE streaming queries
│   │       └── router.py    # Router aggregation
│   ├── core/
│   │   ├── logging.py       # Structured JSON logging
│   │   ├── security.py      # Optional API key verification
│   │   └── exceptions.py   # Custom domain exceptions
│   ├── db/
│   │   ├── session.py       # Async SQLAlchemy session factory
│   │   └── models.py        # Document & DocumentChunk with HNSW / GIN indices
│   ├── services/
│   │   ├── chunking.py      # Semantic paragraph & sentence text chunker
│   │   ├── embedding.py     # Gemini text-embedding-004 adapter
│   │   ├── llm.py           # Gemini 1.5 Flash streaming adapter
│   │   ├── retrieval.py     # Hybrid search + Reciprocal Rank Fusion
│   │   ├── cache.py         # Redis query caching service
│   │   └── ingestion.py     # Chunking & vector storage pipeline
│   ├── config.py            # Pydantic v2 application settings
│   └── main.py              # FastAPI application & lifespan management
├── tests/
│   ├── conftest.py          # Pytest async fixtures & service mocks
│   ├── test_chunking.py     # Text chunking unit tests
│   ├── test_hybrid_retrieval.py # RRF mathematical verification
│   └── test_api.py          # API route tests
├── pyproject.toml           # Project configuration & dependencies
└── README.md
```

---

## 🚀 Quickstart (Docker Compose)

The easiest way to run the entire stack (API, PostgreSQL with pgvector, and Redis):

### 1. Clone & Configure Environment
```bash
git clone https://github.com/your-username/rag-backend-service.git
cd rag-backend-service
cp .env.example .env
```

Open `.env` and add your **Google Gemini API Key**:
```ini
GEMINI_API_KEY="AIzaSy...your-gemini-key"
```

### 2. Start the Stack
```bash
docker compose -f docker/docker-compose.yml up --build -d
```

The service will be available at:
* **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative API Docs (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check**: [http://localhost:8000/api/v1/health/ready](http://localhost:8000/api/v1/health/ready)

### 3. Seed Sample Knowledge Base
```bash
python scripts/seed_data.py
```

---

## 🛠 Local Development Setup

If you prefer running Python locally:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies with development tools
pip install -e ".[dev]"

# 3. Start database and redis via docker
docker compose -f docker/docker-compose.yml up -d db redis

# 4. Run the API server with hot-reload
uvicorn src.main:app --reload --port 8000
```

---

## 📡 API Reference & Usage

### 1. Ingest a Document
`POST /api/v1/documents`
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "PostgreSQL pgvector Extension",
    "source": "https://github.com/pgvector/pgvector",
    "content": "pgvector is an open-source vector similarity search extension for PostgreSQL. It supports L2 distance, inner product, and cosine distance."
  }'
```

### 2. Query Knowledge Base (Standard)
`POST /api/v1/query`
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What distance metrics does pgvector support?",
    "use_cache": true
  }'
```
**Response:**
```json
{
  "query": "What distance metrics does pgvector support?",
  "answer": "Based on the provided documentation, pgvector supports L2 distance, inner product, and cosine distance [Source 1].",
  "cached": false,
  "latency_ms": 382.4,
  "citations": [
    {
      "document_title": "PostgreSQL pgvector Extension",
      "chunk_index": 0,
      "content": "pgvector is an open-source vector similarity search extension for PostgreSQL. It supports L2 distance, inner product, and cosine distance.",
      "score": 0.03279
    }
  ]
}
```

### 3. Real-Time Streaming Query (SSE)
`GET /api/v1/query/stream?q=...`
```bash
curl -N "http://localhost:8000/api/v1/query/stream?q=What+is+Reciprocal+Rank+Fusion"
```
**Output Stream:**
```
data: {"token": "Reciprocal "}

data: {"token": "Rank "}

data: {"token": "Fusion "}

data: {"token": "(RRF) "}

...
data: {"event": "citations", "citations": [...]}

data: [DONE]
```

---

## 🧪 Testing & Quality Assurance

Run the comprehensive automated test suite:

```bash
# Run tests with code coverage
pytest -v --cov=src tests/

# Run linter and formatting checks
ruff check src tests
ruff format --check src tests
```

---

## 📄 License
MIT License. Free for commercial and non-commercial use.
