import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.core.logging import setup_logging
from src.core.exceptions import RAGException
from src.db.session import init_db
from src.services.cache import get_cache_service
from src.api.middleware import ObservabilityMiddleware
from src.api.v1.router import api_v1_router

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure logging and database
    setup_logging(debug=settings.DEBUG)
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    try:
        await init_db()
        from src.db import session as db_session
        db_session.is_db_connected = True
        logger.info("Database initialized with pgvector extension.")
    except Exception as e:
        from src.db import session as db_session
        db_session.is_db_connected = False
        logger.warning(
            f"Could not connect to database on startup: {e}. Running in in-memory fallback mode."
        )

    yield

    # Shutdown: cleanly close redis connection
    logger.info("Shutting down application...")
    cache = get_cache_service()
    await cache.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="High-performance RAG backend service featuring hybrid pgvector search, Gemini LLM streaming, and Redis caching.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Custom Observability Middleware (Tracing & Latency Headers)
app.add_middleware(ObservabilityMiddleware)

# 3. Global Exception Handlers
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": type(exc).__name__, "message": exc.message},
    )


# 4. Mount API Routes
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health/ready",
    }
