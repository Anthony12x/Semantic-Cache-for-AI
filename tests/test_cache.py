"""Tests for the semantic cache service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.cache import SemanticCache


class TestSemanticCacheSearch:
    """Tests for the cache search functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_redis_client, sample_cache_hit):
        """A semantically similar entry should return a cached response."""
        mock_doc = MagicMock()
        mock_doc.vector_score = 0.05
        mock_doc.prompt = sample_cache_hit["prompt"]
        mock_doc.response = sample_cache_hit["response"]

        mock_result = MagicMock()
        mock_result.docs = [mock_doc]

        mock_ft = mock_redis_client.ft.return_value
        mock_ft.search = AsyncMock(return_value=mock_result)

        cache = SemanticCache()
        cache._client = mock_redis_client

        result = await cache.search([0.1] * 384, "What does a reverse proxy do?")

        assert result is not None
        assert result["response"] == sample_cache_hit["response"]
        assert result["score"] == 0.05

    @pytest.mark.asyncio
    async def test_cache_miss_no_docs(self, mock_redis_client):
        """No matching documents should return None."""
        mock_result = MagicMock()
        mock_result.docs = []

        mock_ft = mock_redis_client.ft.return_value
        mock_ft.search = AsyncMock(return_value=mock_result)

        cache = SemanticCache()
        cache._client = mock_redis_client

        result = await cache.search([0.1] * 384, "some query")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_score_above_threshold(self, mock_redis_client, mock_settings):
        """Score above threshold should be rejected."""
        mock_doc = MagicMock()
        mock_doc.vector_score = 0.20
        mock_doc.prompt = "similar text"
        mock_doc.response = "cached response"

        mock_result = MagicMock()
        mock_result.docs = [mock_doc]

        mock_ft = mock_redis_client.ft.return_value
        mock_ft.search = AsyncMock(return_value=mock_result)

        mock_settings.CACHE_THRESHOLD = 0.12
        cache = SemanticCache()
        cache._client = mock_redis_client

        # Temporarily override settings
        original = cache._client
        import app.services.cache as cache_module
        cache_module.settings = mock_settings

        result = await cache.search([0.1] * 384, "similar text")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_query_returns_none(self, mock_redis_client):
        """Empty query text should return None without hitting Redis."""
        cache = SemanticCache()
        cache._client = mock_redis_client

        result = await cache.search([0.1] * 384, "")
        assert result is None

    @pytest.mark.asyncio
    async def test_length_variance_rejection(self, mock_redis_client):
        """Large length difference should reject the cache hit."""
        mock_doc = MagicMock()
        mock_doc.vector_score = 0.01
        mock_doc.prompt = "a" * 1000  # Very long prompt
        mock_doc.response = "response"

        mock_result = MagicMock()
        mock_result.docs = [mock_doc]

        mock_ft = mock_redis_client.ft.return_value
        mock_ft.search = AsyncMock(return_value=mock_result)

        cache = SemanticCache()
        cache._client = mock_redis_client

        result = await cache.search([0.1] * 384, "short")
        assert result is None

    @pytest.mark.asyncio
    async def test_close_closes_connection(self, mock_redis_client):
        """close() should call aclose on the Redis client."""
        cache = SemanticCache()
        cache._client = mock_redis_client

        await cache.close()
        mock_redis_client.aclose.assert_called_once()
