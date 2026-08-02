import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from redis.exceptions import ConnectionError, TimeoutError

from app.api.v1.routes.metrics import metrics
from app.core.guardrails import GuardrailManager
from app.models.dto import QueryRequest, QueryResponse
from app.services.cache import SemanticCache
from app.services.embedder import ColbertEmbeddingEngine, EmbedderFactory
from app.services.llm import LLMRouter

logger = logging.getLogger(__name__)

router = APIRouter()
embedder = EmbedderFactory.create()

try:
    colbert = ColbertEmbeddingEngine()
except RuntimeError as e:
    logger.warning("ColBERT engine disabled: %s", e)
    colbert = None


cache = SemanticCache()
guardrails = GuardrailManager()


@router.post("/generate", response_model=QueryResponse)
async def process_prompt(request: QueryRequest, background_tasks: BackgroundTasks):
    """Two-stage cache lookup (BGE → ColBERT MaxSim), falls back to LLM on miss."""
    start_time = time.perf_counter()
    await metrics.inc("requests_total")

    if colbert:
        vector, colbert_matrix = await asyncio.gather(
            embedder.get_embedding_async(request.prompt),
            colbert.get_colbert_embedding_async(request.prompt),
        )
    else:
        vector = await embedder.get_embedding_async(request.prompt)
        colbert_matrix = None

    try:
        cached_data = await cache.search(vector, request.prompt, query_colbert=colbert_matrix)
    except (ConnectionError, TimeoutError):
        logger.warning("Redis connection failed, skipping cache search")
        cached_data = None

    if cached_data:
        latency = (time.perf_counter() - start_time) * 1000
        await metrics.inc("cache_hits_total")

        # Estimate tokens saved (1 token ≈ 4 characters)
        chars_saved = len(request.prompt) + len(cached_data["response"])
        await metrics.inc("token_savings_est_total", chars_saved // 4)

        await metrics.observe_latency(latency / 1000)
        return QueryResponse(cached=True, response=cached_data["response"], latency_ms=latency)

    try:
        await guardrails.check_circuit()

        llm = LLMRouter.get_provider(request.provider)
        llm_response = await llm.generate(
            request.prompt, request.model, request.temperature, request.api_key
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        # Re-raise guardrail HTTP exceptions directly
        raise
    except Exception:
        await metrics.inc("llm_errors_total")
        await guardrails.record_error()
        logger.exception(
            "LLM generation failed for provider=%s model=%s", request.provider, request.model
        )
        raise HTTPException(
            status_code=502, detail="Inference failed. Check server logs for details."
        )

    async def safe_cache_store(*args):
        try:
            await cache.store(*args)
        except (ConnectionError, TimeoutError):
            logger.warning("Redis connection failed, skipping cache store")

    prompt_id = str(uuid.uuid4())
    background_tasks.add_task(
        safe_cache_store, prompt_id, request.prompt, llm_response, vector, colbert_matrix
    )

    latency = (time.perf_counter() - start_time) * 1000
    await metrics.inc("cache_misses_total")
    await metrics.observe_latency(latency / 1000)
    return QueryResponse(cached=False, response=llm_response, latency_ms=latency)
