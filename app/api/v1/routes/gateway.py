from fastapi import APIRouter, HTTPException, BackgroundTasks
import time
import uuid
import aiohttp
from app.models.dto import QueryRequest, QueryResponse
from app.services.embedder import LocalEmbedder
from app.services.cache import SemanticCache
from app.core.config import settings

router = APIRouter()
embedder = LocalEmbedder()
cache = SemanticCache()

async def generate_from_ollama(prompt: str, model: str, temperature: float) -> str:
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "stream": False}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{settings.OLLAMA_URL}/api/generate", json=payload) as resp:
            if resp.status != 200:
                raise Exception(f"Ollama returned {resp.status}")
            data = await resp.json()
            return data["response"]

@router.post("/generate", response_model=QueryResponse)
async def process_prompt(request: QueryRequest, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    vector = embedder.get_embedding(request.prompt)
    cached_data = await cache.search(vector, request.prompt)
    
    if cached_data:
        latency = (time.perf_counter() - start_time) * 1000
        return QueryResponse(cached=True, response=cached_data["response"], latency_ms=latency)
        
    try:
        llm_response = await generate_from_ollama(request.prompt, request.model, request.temperature)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Inference Failure: {str(e)}")
        
    prompt_id = str(uuid.uuid4())
    background_tasks.add_task(cache.store, prompt_id, request.prompt, llm_response, vector)
    
    latency = (time.perf_counter() - start_time) * 1000
    return QueryResponse(cached=False, response=llm_response, latency_ms=latency)