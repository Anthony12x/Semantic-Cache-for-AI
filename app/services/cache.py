import redis.asyncio as redis
from redis.exceptions import ResponseError
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
import numpy as np
from app.core.config import settings

class SemanticCache:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        self.index_name = "idx:prompts"

    async def initialize_schema(self):
        try:
            await self.redis_client.ft(self.index_name).info()
        except ResponseError:
            schema = (
                TextField("prompt"),
                TextField("response"),
                VectorField("embedding", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": settings.VECTOR_DIMENSION,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,             
                    "EF_CONSTRUCTION": 200 
                })
            )
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            await self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)

    async def search(self, query_vector: list[float], query_text: str) -> dict | None:
        """Executes an ANN search with lexical length penalization."""
        vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        query = (
            Query("*=>[KNN 1 @embedding $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("vector_score", "prompt", "response")
            .dialect(2)
        )
        
        results = await self.redis_client.ft(self.index_name).search(query, query_params={"vec": vector_bytes})
        
        if results.docs:
            doc = results.docs[0]
            score = float(doc.vector_score)
            
            # 1. Primary Check: Spatial Vector Distance
            if score <= settings.CACHE_THRESHOLD:
                cached_prompt = doc.prompt
                
                # 2. Secondary Check: Lexical Variance Penalization
                # Calculates the absolute percentage difference in character length
                max_len = max(len(query_text), len(cached_prompt))
                if max_len == 0:
                    return None
                    
                length_variance = abs(len(query_text) - len(cached_prompt)) / max_len
                
                # If the length differs by more than 25%, reject the semantic match
                if length_variance <= 0.25:
                    print(f"[CACHE HIT] Score: {score:.3f} | Variance: {length_variance:.2f}")
                    return {"prompt": cached_prompt, "response": doc.response, "score": score}
                else:
                    print(f"[CACHE REJECTED] Score: {score:.3f} passed, but Length Variance: {length_variance:.2f} exceeded 0.25")
                    
        return None

    async def store(self, prompt_id: str, prompt: str, response: str, embedding: list[float]):
        mapping = {
            "prompt": prompt,
            "response": response,
            "embedding": np.array(embedding, dtype=np.float32).tobytes()
        }
        await self.redis_client.hset(f"cache:{prompt_id}", mapping=mapping)