from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for the ``POST /api/v1/generate`` endpoint."""

    prompt: str = Field(..., description="The user query to process")
    provider: str = Field(
        default="ollama", description="LLM Provider to use (ollama, openai, anthropic)"
    )
    api_key: str | None = Field(
        default=None, description="Client-provided bearer token (for OpenAI/Anthropic)"
    )
    model: str = Field(default="tinyllama", description="Model name specific to the provider")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic)",
    )


class QueryResponse(BaseModel):
    """Response returned by the gateway."""

    cached: bool = Field(description="True if the response was served from cache")
    response: str = Field(description="The generated or cached response text")
    latency_ms: float = Field(description="Total request latency in milliseconds")
