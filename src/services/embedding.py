import hashlib
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import httpx

from src.config import get_settings
from src.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseEmbeddingService(ABC):
    """Abstract base class for text embedding generation."""

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embeds a single search query."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document chunks in batches."""
        pass


class GeminiEmbeddingService(BaseEmbeddingService):
    """Embedding service using Google Gemini Generative Language REST API."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dimension: int = 768):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def _embed_single(self, client: httpx.AsyncClient, text: str) -> List[float]:
        url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "content": {"parts": [{"text": text}]},
            "output_dimensionality": self.dimension,
        }
        resp = await client.post(url, json=payload, timeout=20.0)
        if resp.status_code != 200:
            raise EmbeddingError(f"Gemini API returned {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["embedding"]["values"]

    async def embed_query(self, text: str) -> List[float]:
        try:
            async with httpx.AsyncClient() as client:
                return await self._embed_single(client, text)
        except Exception as e:
            logger.error(f"Failed to generate query embedding via Gemini: {e}")
            raise EmbeddingError(f"Gemini embedding error: {e}") from e

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient() as client:
                results = []
                for t in texts:
                    emb = await self._embed_single(client, t)
                    results.append(emb)
                return results
        except Exception as e:
            logger.error(f"Failed to generate document embeddings via Gemini: {e}")
            raise EmbeddingError(f"Gemini embedding error: {e}") from e


class MockEmbeddingService(BaseEmbeddingService):
    """
    Deterministic pseudo-embedding generator (768 dimensions).
    Used for unit tests and local execution when GEMINI_API_KEY is not yet set.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _generate_vector(self, text: str) -> List[float]:
        raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(self.dimension):
            byte_val = raw_hash[i % len(raw_hash)]
            vector.append((byte_val / 255.0) * 2.0 - 1.0)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]


def get_embedding_service() -> BaseEmbeddingService:
    """Factory dependency returning the configured embedding service."""
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip():
        logger.info("Using GeminiEmbeddingService with configured API key.")
        return GeminiEmbeddingService(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
        )
    logger.warning("No GEMINI_API_KEY provided. Defaulting to MockEmbeddingService for offline testing.")
    return MockEmbeddingService(dimension=settings.EMBEDDING_DIMENSION)
