import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "health" in data


@pytest.mark.asyncio
async def test_liveness_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_observability_middleware_headers(async_client: AsyncClient):
    response = await async_client.get("/")
    assert "x-correlation-id" in response.headers
    assert "x-process-time-ms" in response.headers
    # Verify process time is a valid float
    process_time = float(response.headers["x-process-time-ms"])
    assert process_time >= 0.0
