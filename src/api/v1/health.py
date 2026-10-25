from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.services.cache import get_cache_service, CacheService

router = APIRouter(prefix="/health", tags=["Health & Observability"])


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """Liveness probe indicating the web application process is responsive."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness_probe(
    db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
):
    """Readiness probe checking database and redis dependencies."""
    health_status = {
        "status": "ready",
        "database": "unknown",
        "redis": "unknown",
    }

    # 1. Check PostgreSQL + pgvector
    try:
        result = await db.execute(text("SELECT 1;"))
        if result.scalar() == 1:
            health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"unhealthy: {e}"
        health_status["status"] = "degraded"

    # 2. Check Redis
    try:
        redis_client = await cache.get_client()
        if redis_client:
            pong = await redis_client.ping()
            health_status["redis"] = "healthy" if pong else "unhealthy"
        else:
            health_status["redis"] = "disabled"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {e}"
        health_status["status"] = "degraded"

    http_status = (
        status.HTTP_200_OK
        if health_status["status"] == "ready"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=http_status, content=health_status)
