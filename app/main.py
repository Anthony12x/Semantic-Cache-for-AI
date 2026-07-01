from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.routes import gateway
from app.services.cache import SemanticCache

# Configure logging so all modules share a consistent format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Initialise the vector cache index on startup and clean up on shutdown."""
    cache = SemanticCache()
    try:
        await cache.initialize_schema()
    except Exception:
        logging.exception("Failed to initialise Redis cache — gateway will not cache")
        raise

    yield

    # Clean up Redis connection on shutdown
    try:
        await cache.close()
    except Exception:
        logger.warning("Error closing Redis connection on shutdown", exc_info=True)


app = FastAPI(
    title="Semantic Gateway",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(gateway.router, prefix="/api/v1", tags=["gateway"])
