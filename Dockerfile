FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/.cache/huggingface \
    && addgroup --system gatewaygroup \
    && adduser --system --group gatewayuser

COPY --from=builder /install /usr/local
COPY ./app ./app

RUN python -c "\
from huggingface_hub import hf_hub_download; \
from transformers import AutoTokenizer, AutoModel; \
hf_hub_download('BAAI/bge-small-en-v1.5', 'onnx/model.onnx', token=False); \
AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', token=False); \
AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0', token=False); \
AutoModel.from_pretrained('colbert-ir/colbertv2.0', token=False); \
hf_hub_download('colbert-ir/colbertv2.0', 'model.safetensors', token=False);"

RUN chown -R gatewayuser:gatewaygroup /app/.cache/huggingface

USER gatewayuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
