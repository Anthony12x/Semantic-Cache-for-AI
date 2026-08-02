from __future__ import annotations

import abc
import asyncio
import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from app.core.config import settings
from app.core.http import HttpClientManager

logger = logging.getLogger(__name__)

ONNX_MODEL_FILE = "onnx/model.onnx"
TOKENIZER_MAX_LENGTH = 512
ONNX_INTRA_OP_THREADS = 2
POOLING_EPSILON = 1e-9


class EmbeddingEngine(abc.ABC):
    """Interface for all embedding providers."""

    @abc.abstractmethod
    async def get_embedding_async(self, text: str) -> list[float]:
        pass


class LocalEmbeddingEngine(EmbeddingEngine):
    """Generates dense embeddings locally via ONNX Runtime.
    Uses token=False on all HF calls to avoid broken local auth tokens.
    """

    def __init__(self, model_id: str | None = None) -> None:
        model_id = model_id or settings.EMBEDDING_MODEL_ID
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=False)

        logger.info("Downloading/verifying ONNX weights for %s ...", model_id)
        onnx_path = hf_hub_download(repo_id=model_id, filename=ONNX_MODEL_FILE, token=False)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = ONNX_INTRA_OP_THREADS
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        model_dim = self._session.get_outputs()[0].shape[-1]
        if model_dim != settings.VECTOR_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: ONNX model outputs {model_dim}d, "
                f"but VECTOR_DIMENSION is set to {settings.VECTOR_DIMENSION}. "
                f"Update VECTOR_DIMENSION in your .env file."
            )

        logger.info("Embedder initialised (%s, %d-dim)", model_id, model_dim)

    def get_embedding(self, text: str) -> list[float]:
        """Synchronous embedding for a single text string."""
        if not text:
            return [0.0] * settings.VECTOR_DIMENSION

        tokens = self._tokenize(text)
        onnx_inputs = self._build_onnx_inputs(tokens)
        outputs = self._session.run(None, onnx_inputs)
        return self._mean_pool(outputs, tokens["attention_mask"]).tolist()

    async def get_embedding_async(self, text: str) -> list[float]:
        """Offloads ONNX inference to a thread."""
        return await asyncio.to_thread(self.get_embedding, text)

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        return self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors="np",
        )

    @staticmethod
    def _build_onnx_inputs(tokens: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in tokens:
            inputs["token_type_ids"] = tokens["token_type_ids"].astype(np.int64)
        return inputs

    @staticmethod
    def _mean_pool(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean-pools token embeddings with attention mask, then L2-normalizes."""
        token_embeddings = model_output[0]
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=POOLING_EPSILON, a_max=None)
        raw_vector = sum_embeddings / sum_mask

        norm = np.linalg.norm(raw_vector, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return (raw_vector / norm).flatten()


class EnterpriseEmbeddingEngine(EmbeddingEngine):
    """Generates embeddings via OpenAI's embeddings API."""

    def __init__(self, model_id: str = "text-embedding-3-small"):
        self._model_id = model_id
        self._api_key = settings.OPENAI_API_KEY
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env to use OpenAI embeddings")

    async def get_embedding_async(self, text: str) -> list[float]:
        """Calls OpenAI embeddings endpoint and validates dimension."""
        if not text:
            return [0.0] * settings.VECTOR_DIMENSION

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"input": text, "model": self._model_id}

        session = HttpClientManager.get_session()
        async with session.post(
            "https://api.openai.com/v1/embeddings", headers=headers, json=payload
        ) as resp:
            if resp.status != 200:
                logger.error("OpenAI Embedding API returned status %d", resp.status)
                raise RuntimeError(f"OpenAI Embedding API returned status {resp.status}")
            data = await resp.json()

        vector = data["data"][0]["embedding"]

        if len(vector) != settings.VECTOR_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: OpenAI returned {len(vector)}d, "
                f"but VECTOR_DIMENSION is {settings.VECTOR_DIMENSION}"
            )

        return vector


class EmbedderFactory:
    """Resolves EMBEDDING_PROVIDER to the correct engine instance."""

    @staticmethod
    def create() -> EmbeddingEngine:
        provider = settings.EMBEDDING_PROVIDER.lower()
        if provider == "onnx":
            return LocalEmbeddingEngine()
        elif provider == "openai":
            return EnterpriseEmbeddingEngine()
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


class ColbertEmbeddingEngine:
    """Late-interaction reranker. Produces per-token 128d matrices and scores via MaxSim.
    Uses token=False on all HF calls.
    """

    def __init__(self, model_id: str | None = None) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise RuntimeError("ColBERT requires torch and transformers. Please pip install them.")

        self.model_id = model_id or settings.COLBERT_MODEL_ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading ColBERT model %s on %s...", self.model_id, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=False)
        self.q_marker = "[unused0]"
        self.model = AutoModel.from_pretrained(self.model_id, token=False)

        self.colbert_dim = 128
        self.linear = nn.Linear(self.model.config.hidden_size, self.colbert_dim, bias=False)

        try:
            import safetensors.torch
            from huggingface_hub import hf_hub_download

            state_dict_path = hf_hub_download(self.model_id, "model.safetensors", token=False)
            state_dict = safetensors.torch.load_file(state_dict_path)
            if "linear.weight" in state_dict:
                self.linear.weight.data = state_dict["linear.weight"].clone()
        except Exception:
            try:
                import torch
                from huggingface_hub import hf_hub_download

                state_dict_path = hf_hub_download(self.model_id, "pytorch_model.bin", token=False)
                state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
                if "linear.weight" in state_dict:
                    self.linear.weight.data = state_dict["linear.weight"].clone()
            except Exception as e:
                logger.warning("Could not load ColBERT linear layer weights: %s", e)

        self.model.to(self.device)
        self.linear.to(self.device)

        self.model.eval()
        self.linear.eval()

    def get_colbert_embedding(self, text: str) -> np.ndarray:
        """Tokenizes text, runs through BERT + linear projection,
        returns L2-normalized token matrix.
        """
        if not text:
            return np.zeros((1, self.colbert_dim), dtype=np.float32)

        import torch

        text = f"{self.q_marker} {text}"

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = self.linear(outputs.last_hidden_state)
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
            token_embeddings = token_embeddings.cpu().numpy()[0]

        return token_embeddings.astype(np.float32)

    async def get_colbert_embedding_async(self, text: str) -> np.ndarray:
        """Offloads ColBERT inference to a thread."""
        import asyncio

        return await asyncio.to_thread(self.get_colbert_embedding, text)

    @staticmethod
    def compute_maxsim(query_matrix: np.ndarray, doc_matrix: np.ndarray) -> float:
        """Computes MaxSim between two token matrices — sum of per-query-token max dot products."""
        scores = np.dot(query_matrix, doc_matrix.T)
        max_scores = np.max(scores, axis=1)
        return float(np.sum(max_scores))
