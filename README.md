# Semantic Gateway

Cuts LLM costs by caching semantically identical queries — no matter how they're phrased.

You ask "How do I restart a docker container?" once. You ask "How do I reboot a docker instance?" next time. Same answer, zero inference cost, much faster response.

That's it. Just a FastAPI gateway, Redis with HNSW, and an ONNX model running on your CPU.

---

## How It Works

![System Architecture](assets/architecture.png)

**The cache validation.**

A raw vector search will match semantically similar but intentionally different queries ("add two numbers" vs "multiply two numbers"). In enterprise environments, serving a False Positive is a catastrophic failure. It is strictly better to accept a Cache Miss (and pay for the GPU compute) than to serve the wrong context. To guarantee this, the gateway implements two validations for cache:

1. **Cosine distance** via HNSW ANN (Redis). Must be within `CACHE_THRESHOLD`.
2. **Length variance** — If the character counts differ by more than 25%, reject. Short queries and long paraphrases shouldn't match even if the embedding says they're close.

The threshold is calibrated per a dataset using `scripts/calibrate_thresholds.py` — it sweeps cosine distance values and reports F1-score with false-positive detection.

---

## Tech Stack

| Layer | What |
|---|---|
| API | FastAPI + uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via ONNX Runtime |
| Vector Store | Redis Stack (RediSearch HNSW) |
| Inference | Ollama (host-native) |

---

## Prerequisites

- Python 3.11+
- [Docker & Docker Compose](https://docs.docker.com/get-docker/)
- [Ollama](https://ollama.ai/) (installed on the host, not in Docker)

---

## Setup

### 1. Docker infrastructure

```bash
docker compose up -d
```

This spins up the gateway (port 8000) and Redis Stack (port 6379).

### 2. Ollama on the host

Ollama runs natively on your machine (not in Docker) so it can access your GPU directly. Run the provisioning script to verify the daemon and download the weights:

```bash
.\scripts\init_host.ps1
```

### 3. Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings. The defaults should work out of the box.

### 4. Install dependencies

```bash
pip install -e ".[dev]"
```

### 5. Calibrate (optional but recommended)

```bash
python scripts/calibrate_thresholds.py
```

This runs your embedding model against a dataset of positive pairs (should match) and negative pairs (must not match), then outputs a threshold sweep with F1-scores. Pick the threshold with zero false positives and the highest F1. Update `CACHE_THRESHOLD` in `.env`.

---

## Configuration

| Variable | Default |
|---|---|
| `REDIS_URL` | `redis://localhost:6379` |
| `OLLAMA_URL` | `http://localhost:11434`|
| `CACHE_THRESHOLD` | `0.12` |
| `VECTOR_DIMENSION` | `384` |

---

## API

### POST `/api/v1/generate`

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the primary function of a reverse proxy?", "model": "tinyllama"}'
```

Response:

```json
{
  "cached": true,
  "response": "A reverse proxy sits in front of backend servers...",
  "latency_ms": 0.8
}
```

| Field | Meaning |
|---|---|
| `cached` | `true` = served from cache, `false` = fetched from Ollama |
| `response` | The response text |
| `latency_ms` | Total request time. Cache hits are typically <1ms. |

---

## Testing

50 basic prompt pairs across 5 domains (IT helpdesk, software engineering, DevOps, data science, HR/legal). Each pair tests that a rephrased query hits the cache after the original was asked:

```bash
docker exec semantic_cache_db redis-cli FLUSHALL
npx promptfoo@latest eval --no-cache -j 1
```

---

## Project Structure

```
.
├── app/
│   ├── api/v1/routes/gateway.py   # HTTP endpoints
│   ├── core/config.py             # Settings from environment
│   ├── models/dto.py              # Request/response schemas
│   ├── services/
│   │   ├── cache.py               # Redis-backed semantic cache
│   │   └── embedder.py            # ONNX embedding model
│   └── main.py                    # FastAPI entrypoint
├── scripts/
|   ├── init_host.ps1              # Host-level Ollama provisioning
│   └── calibrate_thresholds.py    # Threshold calibration utility
├── tests/
│   ├── conftest.py                # Shared fixtures
│   └── test_cache.py              # Cache service tests
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## What's Next (Not in v1)

- [ ] Switching to a higher-capacity local embedding model for better threshold and more cache hits
- [ ] Cache TTL / eviction policy
- [ ] Externalize validation datasets via CSV ingestion for scalable testing
- [ ] Automate Continuous Calibration pipeline using historical prompt logs for more accurate cache threshold
- [ ] Include Enterprise AI/API support
- [ ] Retry logic on Ollama failures
- [ ] Async embedding to avoid blocking the event loop
---

## License

MIT — see [LICENSE](LICENSE).
