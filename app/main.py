from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routes import gateway, metrics
from app.core.config import settings
from app.core.http import HttpClientManager
from app.core.tracing import TracingMiddleware, request_id_ctx
from app.services.cache import SemanticCache

logger = logging.getLogger(__name__)


def configure_logging():
    """Sets up logging based on LOG_FORMAT and LOG_LEVEL from settings."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if settings.LOG_FORMAT.lower() == "json":

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps(
                    {
                        "ts": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "request_id": request_id_ctx.get(),
                        "msg": record.getMessage(),
                    }
                )

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(level)

        # Force Uvicorn to use our JSON handler
        for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers = [handler]
            uvicorn_logger.propagate = False
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


configure_logging()

_cache = SemanticCache()

# Avoid circular dependencies
from app.api.v1.routes.gateway import guardrails  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialises Redis index on startup, tears down connections on shutdown."""
    await HttpClientManager.start()
    try:
        await _cache.initialize_schema()
    except Exception:
        logger.warning(
            "Failed to initialise Redis cache. Gateway will run without cache.", exc_info=True
        )

    yield

    try:
        await guardrails.close()
        await _cache.close()
    except Exception:
        logger.warning("Error closing Redis connection on shutdown", exc_info=True)

    await HttpClientManager.stop()


app = FastAPI(
    title="Semantic Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

if settings.CORS_ALLOWED_ORIGINS:
    origins = [orig.strip() for orig in settings.CORS_ALLOWED_ORIGINS.split(",") if orig.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(TracingMiddleware)

app.include_router(gateway.router, prefix="/api/v1", tags=["gateway"])
app.include_router(metrics.router, tags=["observability"])


@app.get("/health")
async def health_check():
    """Deep readiness probe — verifies Redis, Embedder, and Ollama."""
    components = {}
    is_healthy = True

    try:
        await _cache.redis_client.ping()
        components["redis"] = "connected"
    except Exception:
        components["redis"] = "disconnected"
        is_healthy = False

    if gateway.embedder is not None:
        components["embedder"] = "loaded"
    else:
        components["embedder"] = "unloaded"
        is_healthy = False

    try:
        session = HttpClientManager.get_session()
        async with session.head(f"{settings.OLLAMA_URL}/api/tags", timeout=2.0) as resp:
            if resp.status == 200:
                components["ollama"] = "reachable"
            else:
                components["ollama"] = f"unreachable ({resp.status})"
                is_healthy = False
    except Exception:
        components["ollama"] = "unreachable"
        is_healthy = False

    status = "healthy" if is_healthy else "degraded"

    if components["redis"] == "disconnected":
        return JSONResponse({"status": "unhealthy", "components": components}, status_code=503)

    return JSONResponse({"status": status, "components": components})
