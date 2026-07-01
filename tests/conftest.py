"""Shared pytest fixtures for the Semantic Gateway test suite."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_redis_client():
    """Return a mock Redis async client with a mock FT (RediSearch) interface."""
    mock_ft = AsyncMock()
    mock_ft.info = AsyncMock()
    mock_ft.create_index = AsyncMock()
    mock_ft.search = AsyncMock()

    mock_client = AsyncMock()
    mock_client.ft.return_value = mock_ft
    mock_client.hset = AsyncMock()
    mock_client.aclose = AsyncMock()

    return mock_client


@pytest.fixture
def mock_settings():
    """Return a mock Settings object with default values."""
    settings = MagicMock()
    settings.REDIS_URL = "redis://localhost:6379"
    settings.OLLAMA_URL = "http://localhost:11434"
    settings.CACHE_THRESHOLD = 0.12
    settings.VECTOR_DIMENSION = 384
    return settings


@pytest.fixture
def mock_embedding():
    """Return a sample 384-dim embedding vector (matches all-MiniLM-L6-v2)."""
    return np.random.randn(384).astype(np.float32).tolist()


@pytest.fixture
def sample_request():
    """Return a valid QueryRequest payload."""
    return {
        "prompt": "What is the primary function of a reverse proxy?",
        "model": "tinyllama",
        "temperature": 0.0,
    }


@pytest.fixture
def sample_cache_hit():
    """Return a mock cache hit response."""
    return {
        "prompt": "What does a reverse proxy do in a network?",
        "response": "A reverse proxy sits in front of backend servers...",
        "score": 0.05,
    }


@pytest.fixture
def sample_cache_miss():
    """Return a mock cache miss (None)."""
    return None
