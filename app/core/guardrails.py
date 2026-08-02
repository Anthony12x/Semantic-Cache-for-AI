import logging

import redis.asyncio as redis
from fastapi import HTTPException
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)


class GuardrailManager:
    """Provides Redis-backed API budget protection: Rate Limiting and Circuit Breaking."""

    def __init__(self):
        self.redis_client = redis.from_url(
            settings.REDIS_URL, decode_responses=False, health_check_interval=1
        )
        self.cb_errors = settings.LLM_CIRCUIT_BREAKER_ERRORS
        self.cb_window = settings.LLM_CIRCUIT_BREAKER_WINDOW

    async def check_circuit(self):
        """Checks if the circuit breaker is open (tripped)."""
        if self.cb_errors <= 0:
            return

        key = "guardrails:circuitbreaker"
        try:
            raw_count = await self.redis_client.get(key)
        except (ConnectionError, TimeoutError):
            logger.warning("Redis unreachable, skipping circuit breaker check")
            return

        if raw_count:
            count = int(raw_count)
            if count >= self.cb_errors:
                logger.warning("Circuit breaker is OPEN. Rejecting request.")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Service is temporarily unavailable due to high error rates. "
                        "Please try again later."
                    ),
                )

    async def record_error(self):
        """Increments the circuit breaker error counter."""
        if self.cb_errors <= 0:
            return

        key = "guardrails:circuitbreaker"
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            # Only set expiry if it's the first error in this window, otherwise let it accumulate
            pipe.ttl(key)
            results = await pipe.execute()

            current_count = results[0]
            ttl = results[1]

            if ttl == -1 or ttl == -2:
                # -1 means no expiry
                # -2 means key doesn't exist (shouldn't happen immediately after incr)
                await self.redis_client.expire(key, self.cb_window)

            logger.error("Circuit Breaker error recorded: %d/%d", current_count, self.cb_errors)
        except (ConnectionError, TimeoutError):
            logger.warning("Redis unreachable, skipping circuit breaker error record")

    async def close(self):
        if hasattr(self.redis_client, "aclose"):
            await self.redis_client.aclose()
        else:
            await self.redis_client.close()
