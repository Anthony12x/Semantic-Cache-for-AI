from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_MODEL_FILE = "onnx/model.onnx"
TOKENIZER_MAX_LENGTH = 512
ONNX_INTRA_OP_THREADS = 2
POOLING_EPSILON = 1e-9


class LocalEmbedder:
    """Wraps an ONNX embedding model for CPU inference.

    Downloads the model weights and tokenizer on first instantiation,
    then provides a simple ``get_embedding(text)`` interface.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info("Downloading/verifying ONNX weights for %s ...", model_id)
        onnx_path = hf_hub_download(repo_id=model_id, filename=ONNX_MODEL_FILE)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = ONNX_INTRA_OP_THREADS
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Validate that the ONNX model dimension matches the configured vector dimension
        model_dim = self._session.get_outputs()[0].shape[-1]
        if model_dim != settings.VECTOR_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: ONNX model outputs {model_dim}d, "
                f"but VECTOR_DIMENSION is set to {settings.VECTOR_DIMENSION}. "
                f"Update VECTOR_DIMENSION in your .env file."
            )

        logger.info("Embedder initialised (%s, %d-dim)", model_id, model_dim)

    # -- public API ----------------------------------------------------------

    def get_embedding(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*.

        Empty strings are handled gracefully by returning a zero vector.
        """
        if not text:
            logger.debug("get_embedding() skipped: empty text")
            return [0.0] * settings.VECTOR_DIMENSION

        tokens = self._tokenize(text)
        onnx_inputs = self._build_onnx_inputs(tokens)
        outputs = self._session.run(None, onnx_inputs)
        return self._mean_pool(outputs, tokens["attention_mask"]).tolist()

    # -- internals -----------------------------------------------------------

    def _dim(self) -> int:
        """Return the embedding dimension (inferred from the model)."""
        return self._session.get_outputs()[0].shape[-1]

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        """Tokenize *text* and return tensors ready for the ONNX model."""
        return self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors="np",
        )

    @staticmethod
    def _build_onnx_inputs(
        tokens: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Convert tokenised outputs to the ONNX model's expected input format."""
        inputs: dict[str, np.ndarray] = {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        }
        # Some models (e.g. BERT-style) expect token_type_ids.
        if "token_type_ids" in tokens:
            inputs["token_type_ids"] = tokens["token_type_ids"].astype(np.int64)
        return inputs

    @staticmethod
    def _mean_pool(
        model_output: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Mean-pool token embeddings and L2-normalise."""
        token_embeddings = model_output[0]
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=POOLING_EPSILON, a_max=None)
        raw_vector = sum_embeddings / sum_mask

        norm = np.linalg.norm(raw_vector, axis=1, keepdims=True)
        # Guard against zero-norm (shouldn't happen with valid text, but be safe)
        norm = np.where(norm == 0, 1.0, norm)
        return (raw_vector / norm).flatten()
