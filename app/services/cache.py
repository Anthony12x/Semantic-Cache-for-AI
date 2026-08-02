import logging

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.core.config import settings
from app.services.embedder import ColbertEmbeddingEngine

logger = logging.getLogger(__name__)


class SemanticCache:
    """HNSW vector cache backed by Redis Stack.
    Supports two-stage retrieval (BGE + ColBERT MaxSim).
    """

    def __init__(self):
        self.redis_client = redis.from_url(
            settings.REDIS_URL, decode_responses=False, health_check_interval=1
        )
        self.index_name = f"idx:prompts:{settings.EMBEDDING_MODEL_NAME}"

    async def initialize_schema(self):
        """Creates the RediSearch HNSW index if it doesn't already exist."""
        try:
            await self.redis_client.ft(self.index_name).info()
        except ResponseError:
            schema = (
                TextField("prompt"),
                TextField("response"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": settings.VECTOR_DIMENSION,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            )
            definition = IndexDefinition(
                prefix=[f"cache:{settings.EMBEDDING_MODEL_NAME}:"], index_type=IndexType.HASH
            )
            try:
                await self.redis_client.ft(self.index_name).create_index(
                    fields=schema, definition=definition
                )
            except ResponseError as e:
                if "Index already exists" not in str(e):
                    raise

    async def search(
        self, query_vector: list[float], query_text: str, query_colbert: np.ndarray = None
    ) -> dict | None:
        """KNN-5 candidate retrieval via BGE cosine distance,
        reranked by ColBERT MaxSim when available.
        """
        vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        query = (
            Query("*=>[KNN 5 @embedding $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("vector_score", "prompt", "response")
            .dialect(2)
        )

        try:
            results = await self.redis_client.ft(self.index_name).search(
                query, query_params={"vec": vector_bytes}
            )
        except ResponseError:
            await self.initialize_schema()
            return None

        best_hit = None
        best_maxsim = -1.0

        for doc in results.docs:
            score = float(doc.vector_score)
            cached_prompt = (
                doc.prompt
                if isinstance(doc.prompt, str)
                else doc.prompt.decode("utf-8", errors="replace")
            )
            cached_response = (
                doc.response
                if isinstance(doc.response, str)
                else doc.response.decode("utf-8", errors="replace")
            )

            # 1. Exact/Near-Exact match bypass (if BGE distance is extremely small)
            if score <= settings.CACHE_THRESHOLD:
                max_len = max(len(query_text), len(cached_prompt))
                if max_len > 0:
                    length_variance = abs(len(query_text) - len(cached_prompt)) / max_len
                    if length_variance <= 0.25:
                        logger.info(
                            "Cache hit (exact) — score: %.3f, variance: %.2f",
                            score,
                            length_variance,
                        )
                        await self.redis_client.expire(doc.id, settings.CACHE_TTL_SECONDS)
                        return {
                            "prompt": cached_prompt,
                            "response": cached_response,
                            "score": score,
                            "doc_id": doc.id,
                        }

            raw_colbert = await self.redis_client.hget(
                doc.id, b"colbert_embedding" if not isinstance(doc.id, str) else "colbert_embedding"
            )

            if not raw_colbert or query_colbert is None:
                continue

            doc_matrix = np.frombuffer(raw_colbert, dtype=np.float32).reshape(-1, 128)
            maxsim_score = ColbertEmbeddingEngine.compute_maxsim(query_colbert, doc_matrix)

            if maxsim_score >= settings.MAXSIM_THRESHOLD and maxsim_score > best_maxsim:
                best_maxsim = maxsim_score
                best_hit = {
                    "prompt": cached_prompt,
                    "response": doc.response,
                    "score": maxsim_score,
                    "doc_id": doc.id,
                }

        if best_hit:
            logger.info("Cache hit (MaxSim) — score: %.2f", best_hit["score"])
            await self.redis_client.expire(best_hit["doc_id"], settings.CACHE_TTL_SECONDS)
            return best_hit

        return None

    async def store(
        self,
        prompt_id: str,
        prompt: str,
        response: str,
        embedding: list[float],
        colbert_matrix: np.ndarray = None,
    ):
        """Persists a prompt-response pair with its dense and ColBERT embeddings, then sets TTL."""
        mapping = {
            "prompt": prompt,
            "response": response,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        }
        if colbert_matrix is not None:
            mapping["colbert_embedding"] = colbert_matrix.tobytes()

        key = f"cache:{settings.EMBEDDING_MODEL_NAME}:{prompt_id}"
        await self.redis_client.hset(key, mapping=mapping)
        await self.redis_client.expire(key, settings.CACHE_TTL_SECONDS)

    async def close(self):
        """Closes the async Redis connection."""
        if hasattr(self.redis_client, "aclose"):
            await self.redis_client.aclose()
        else:
            await self.redis_client.close()
