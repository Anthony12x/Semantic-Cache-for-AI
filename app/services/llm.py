import abc
import logging

from app.core.config import settings
from app.core.http import HttpClientManager

logger = logging.getLogger(__name__)


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers."""

    @abc.abstractmethod
    async def generate(
        self, prompt: str, model: str, temperature: float, api_key: str | None = None
    ) -> str:
        pass


class OllamaProvider(LLMProvider):
    """Routes generation to a local Ollama instance."""

    async def generate(
        self, prompt: str, model: str, temperature: float, api_key: str | None = None
    ) -> str:
        payload = {"model": model, "prompt": prompt, "temperature": temperature, "stream": False}
        session = HttpClientManager.get_session()
        async with session.post(f"{settings.OLLAMA_URL}/api/generate", json=payload) as resp:
            if resp.status != 200:
                logger.error("Ollama returned status %d", resp.status)
                raise RuntimeError(f"Ollama returned status {resp.status}")
            data = await resp.json()
            return data["response"]


class OpenAIProvider(LLMProvider):
    """Routes generation to the OpenAI chat completions API."""

    def __init__(self):
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env to use OpenAI")
        self._api_key: str = api_key

    async def generate(
        self, prompt: str, model: str, temperature: float, api_key: str | None = None
    ) -> str:
        key_to_use = api_key or self._api_key
        headers = {"Authorization": f"Bearer {key_to_use}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        session = HttpClientManager.get_session()
        async with session.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        ) as resp:
            if resp.status != 200:
                logger.error("OpenAI API returned status %d", resp.status)
                raise RuntimeError(f"OpenAI API returned status {resp.status}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """Routes generation to the Anthropic messages API."""

    def __init__(self):
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in .env to use Anthropic")
        self._api_key: str = api_key

    async def generate(
        self, prompt: str, model: str, temperature: float, api_key: str | None = None
    ) -> str:
        key_to_use = api_key or self._api_key
        headers = {
            "x-api-key": key_to_use,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        session = HttpClientManager.get_session()
        async with session.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=payload
        ) as resp:
            if resp.status != 200:
                logger.error("Anthropic API returned status %d", resp.status)
                raise RuntimeError(f"Anthropic API returned status {resp.status}")
            data = await resp.json()
            return data["content"][0]["text"]


class LLMRouter:
    """Resolves a provider name to an LLMProvider instance."""

    @staticmethod
    def get_provider(provider_name: str) -> LLMProvider:
        name = provider_name.lower()
        if name == "ollama":
            return OllamaProvider()
        elif name == "openai":
            return OpenAIProvider()
        elif name == "anthropic":
            return AnthropicProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
