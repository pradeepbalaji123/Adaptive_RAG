"""Lazy SentenceTransformer loading with normalized cosine embeddings."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .utils import resolve_device


class SentenceEmbeddingRegistry:
    """Reuse compact embedding models without duplicating them in memory."""

    def __init__(self, device: str = "auto", batch_size: int = 64) -> None:
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._models: dict[str, Any] = {}
        self._model_ids: dict[str, str] = {}

    def get(self, key: str, model_id: str) -> Any:
        if key in self._models:
            if self._model_ids[key] != model_id:
                raise ValueError(f"Embedding key {key!r} was already bound to a different model")
            return self._models[key]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers before loading embedding models") from exc
        model = SentenceTransformer(model_id, device=self.device)
        model.eval()
        self._models[key] = model
        self._model_ids[key] = model_id
        return model

    def tokenizer(self, key: str, model_id: str) -> Any:
        return self.get(key, model_id).tokenizer

    def encode(self, key: str, model_id: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty text collection")
        if any(not str(text).strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        model = self.get(key, model_id)
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding model produced NaN or Inf")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Embeddings are not L2 normalized")
        return np.ascontiguousarray(vectors)

