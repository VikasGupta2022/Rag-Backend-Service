import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.services.embedding import MockEmbeddingService, get_embedding_service
from src.services.llm import MockLLMService, get_llm_service
from src.services.cache import CacheService, get_cache_service


@pytest.fixture(scope="session")
def mock_embedding_service():
    return MockEmbeddingService(dimension=768)


@pytest.fixture(scope="session")
def mock_llm_service():
    return MockLLMService()


@pytest.fixture(scope="session")
def mock_cache_service():
    return CacheService(redis_url="", enabled=False)


@pytest_asyncio.fixture
async def async_client(
    mock_embedding_service, mock_llm_service, mock_cache_service
) -> AsyncGenerator[AsyncClient, None]:
    """Test HTTP client with mocked external services."""
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_service
    app.dependency_overrides[get_cache_service] = lambda: mock_cache_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
