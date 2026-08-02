import logging
from collections import defaultdict

import redis.asyncio as redis
from fastapi import APIRouter

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class MetricsStore:
    """Redis-backed counters and histogram for Prometheus-format scraping (multi-worker safe)."""

    def __init__(self):
        # We need our own client so we don't depend on the cache module
        self.redis_client = redis.from_url(
            settings.REDIS_URL, decode_responses=True, health_check_interval=1
        )
        self.latency_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.key = "gateway:metrics"

    async def inc(self, name: str, value: int = 1):
        try:
            await self.redis_client.hincrby(self.key, name, value)
        except Exception:
            logger.warning("Failed to increment metric %s", name, exc_info=True)

    async def observe_latency(self, seconds: float):
        try:
            pipe = self.redis_client.pipeline()
            pipe.hincrbyfloat(self.key, "latency_sum", seconds)
            pipe.hincrby(self.key, "latency_count", 1)

            for bucket in self.latency_buckets:
                if seconds <= bucket:
                    pipe.hincrby(self.key, f"latency_bucket_{bucket:.3f}", 1)
            pipe.hincrby(self.key, "latency_bucket_+Inf", 1)

            await pipe.execute()
        except Exception:
            logger.warning("Failed to observe latency", exc_info=True)

    async def render(self) -> str:
        try:
            raw_metrics = await self.redis_client.hgetall(self.key)
        except Exception:
            logger.warning("Failed to fetch metrics from Redis", exc_info=True)
            raw_metrics = {}

        # Default dictionary to handle missing keys gracefully
        m = defaultdict(float, {k: float(v) for k, v in raw_metrics.items()})

        lines = []

        lines.append("# HELP gateway_requests_total Total requests handled by the gateway.")
        lines.append("# TYPE gateway_requests_total counter")
        lines.append(f"gateway_requests_total {int(m['requests_total'])}")

        lines.append("# HELP gateway_cache_hits_total Requests served from the semantic cache.")
        lines.append("# TYPE gateway_cache_hits_total counter")
        lines.append(f"gateway_cache_hits_total {int(m['cache_hits_total'])}")

        lines.append(
            "# HELP gateway_cache_misses_total Requests that missed the cache and hit the LLM."
        )
        lines.append("# TYPE gateway_cache_misses_total counter")
        lines.append(f"gateway_cache_misses_total {int(m['cache_misses_total'])}")

        lines.append("# HELP gateway_llm_errors_total LLM provider errors.")
        lines.append("# TYPE gateway_llm_errors_total counter")
        lines.append(f"gateway_llm_errors_total {int(m['llm_errors_total'])}")

        lines.append("# HELP gateway_token_savings_est_total Estimated tokens saved by cache hits.")
        lines.append("# TYPE gateway_token_savings_est_total counter")
        lines.append(f"gateway_token_savings_est_total {int(m['token_savings_est_total'])}")

        hits = m["cache_hits_total"]
        misses = m["cache_misses_total"]
        total = hits + misses
        ratio = (hits / total) if total > 0 else 0.0
        lines.append("# HELP gateway_cache_hit_ratio Computed cache hit ratio.")
        lines.append("# TYPE gateway_cache_hit_ratio gauge")
        lines.append(f"gateway_cache_hit_ratio {ratio:.4f}")

        lines.append("# HELP gateway_request_duration_seconds Request latency histogram.")
        lines.append("# TYPE gateway_request_duration_seconds histogram")
        cumulative = 0
        for bucket in self.latency_buckets:
            key = f"latency_bucket_{bucket:.3f}"
            cumulative += int(m[key])
            lines.append(f'gateway_request_duration_seconds_bucket{{le="{bucket}"}} {cumulative}')

        cumulative += int(m["latency_bucket_+Inf"])
        lines.append(f'gateway_request_duration_seconds_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"gateway_request_duration_seconds_sum {m['latency_sum']:.6f}")
        lines.append(f"gateway_request_duration_seconds_count {int(m['latency_count'])}")

        return "\n".join(lines) + "\n"

    async def close(self):
        if hasattr(self.redis_client, "aclose"):
            await self.redis_client.aclose()
        else:
            await self.redis_client.close()


metrics = MetricsStore()


@router.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    rendered = await metrics.render()
    return PlainTextResponse(rendered, media_type="text/plain; version=0.0.4; charset=utf-8")
