"""Exact reference-free R, U, C, normalization, J, and selection logic."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np

from .utils import resolve_device


def compute_retrieval_confidence(similarity_scores: Sequence[float]) -> float:
    """Methodology Module 5: R is mean Top-K cosine similarity."""

    scores = np.asarray(similarity_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Retrieval confidence is undefined for empty retrieval")
    if not np.isfinite(scores).all():
        raise ValueError("Similarity scores contain NaN or Inf")
    return float(scores.mean())


def semantic_uncertainty_from_embeddings(answer_embeddings: np.ndarray) -> tuple[float, float, list[float]]:
    """Methodology Module 6: U = 1 - mean cosine(A1,A2; A1,A3; A2,A3)."""

    vectors = np.asarray(answer_embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != 3:
        raise ValueError("Semantic uncertainty requires embeddings for exactly three answers")
    if not np.isfinite(vectors).all():
        raise ValueError("Answer embeddings contain NaN or Inf")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Answer embedding has zero norm")
    vectors = vectors / norms
    similarities = [
        float(np.dot(vectors[0], vectors[1])),
        float(np.dot(vectors[0], vectors[2])),
        float(np.dot(vectors[1], vectors[2])),
    ]
    semantic_consistency = float(np.mean(similarities))
    uncertainty = 1.0 - semantic_consistency
    if not math.isfinite(uncertainty):
        raise ValueError("Semantic uncertainty is not finite")
    return uncertainty, semantic_consistency, similarities


def estimate_semantic_uncertainty(
    generated_samples: Sequence[str],
    embed_answers: Callable[[Sequence[str]], np.ndarray],
) -> tuple[float, float, list[float]]:
    if len(generated_samples) != 3 or any(not answer.strip() for answer in generated_samples):
        raise ValueError("Exactly three non-empty sampled answers are required")
    return semantic_uncertainty_from_embeddings(embed_answers(generated_samples))


def split_answer_sentences(answer: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]
    return sentences or ([answer.strip()] if answer.strip() else [])


class NLIScorer:
    """Batched DeBERTa NLI entailment probabilities."""

    def __init__(self, model_id: str, device: str = "auto", batch_size: int = 16) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.entailment_index: int | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        id2label = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}
        matches = [index for index, label in id2label.items() if "entail" in label]
        if matches:
            self.entailment_index = matches[0]
        elif self.model_id == "cross-encoder/nli-deberta-v3-base" and self.model.config.num_labels == 3:
            # The official model card defines [contradiction, entailment, neutral].
            self.entailment_index = 1
        else:
            raise RuntimeError(f"Cannot determine entailment label from model config: {id2label}")

    def entailment_probabilities(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=np.float64)
        self.load()
        import torch

        assert self.model is not None and self.tokenizer is not None and self.entailment_index is not None
        probabilities: list[np.ndarray] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            features = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation="only_first",
                max_length=min(int(getattr(self.tokenizer, "model_max_length", 512)), 512),
                return_tensors="pt",
            )
            features = {name: tensor.to(self.device) for name, tensor in features.items()}
            try:
                with torch.inference_mode():
                    logits = self.model(**features).logits
                    values = torch.softmax(logits.float(), dim=-1)[:, self.entailment_index]
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA out of memory during NLI. Set nli_device='cpu' without changing the methodology."
                ) from exc
            probabilities.append(values.detach().cpu().numpy())
        result = np.concatenate(probabilities).astype(np.float64)
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError("Invalid NLI entailment probability")
        return result


def compute_answer_context_consistency(
    answer: str,
    retrieved_chunks: Sequence[str],
    nli_scorer: NLIScorer,
) -> float:
    """Project specification section 15 sentence-by-chunk max, then sentence mean."""

    sentences = split_answer_sentences(answer)
    chunks = [chunk.strip() for chunk in retrieved_chunks if chunk.strip()]
    if not sentences or not chunks:
        raise ValueError("Context consistency requires a non-empty answer and context")
    pairs = [(chunk, sentence) for sentence in sentences for chunk in chunks]
    probabilities = nli_scorer.entailment_probabilities(pairs).reshape(len(sentences), len(chunks))
    score = float(probabilities.max(axis=1).mean())
    if not math.isfinite(score):
        raise ValueError("Context consistency is not finite")
    return score


def min_max_normalize(values: Sequence[float], equal_value: float = 0.5) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Normalization requires finite, non-empty values")
    lower, upper = float(array.min()), float(array.max())
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-12):
        # Equal candidates contain no ranking information, so assign a neutral score.
        return [float(equal_value)] * len(array)
    return [float(value) for value in ((array - lower) / (upper - lower))]


def compute_reference_free_score(
    retrieval_confidence_normalized: float,
    semantic_confidence_normalized: float,
    context_consistency_normalized: float,
) -> float:
    """Methodology Module 8: J = [R' + (1-U)' + C'] / 3."""

    values = np.asarray([
        retrieval_confidence_normalized,
        semantic_confidence_normalized,
        context_consistency_normalized,
    ], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Normalized reference-free signals must be in [0, 1]")
    return float(values.mean())


def normalize_candidate_signals(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize signals only within the current query's candidate set."""

    if not records:
        raise ValueError("No candidate records to normalize")
    query_ids = {record["query_id"] for record in records}
    if len(query_ids) != 1:
        raise ValueError("Signals must be normalized within exactly one query")
    r_norm = min_max_normalize([float(record["retrieval_confidence"]) for record in records])
    semantic_norm = min_max_normalize([float(record["semantic_confidence"]) for record in records])
    c_norm = min_max_normalize([float(record["context_consistency"]) for record in records])
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = deepcopy(record)
        item["retrieval_confidence_normalized"] = r_norm[index]
        item["semantic_confidence_normalized"] = semantic_norm[index]
        item["semantic_uncertainty_normalized"] = 1.0 - semantic_norm[index]
        item["context_consistency_normalized"] = c_norm[index]
        item["combined_score"] = compute_reference_free_score(r_norm[index], semantic_norm[index], c_norm[index])
        item["R_normalized"] = item["retrieval_confidence_normalized"]
        item["U_normalized"] = item["semantic_uncertainty_normalized"]
        item["C_normalized"] = item["context_consistency_normalized"]
        item["J"] = item["combined_score"]
        normalized.append(item)
    return normalized


def select_adaptive_candidate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reference-free argmax J with fixed candidate-order tie-breaking."""

    if not records:
        raise ValueError("Cannot select from an empty candidate collection")
    forbidden = {"gold_answer", "gold_answers", "reference_answer", "reference_answers", "exact_match", "f1"}
    leaked = forbidden.intersection(key for record in records for key in record)
    if leaked:
        raise ValueError(f"Evaluation fields reached adaptive selection: {sorted(leaked)}")
    if len({record["query_id"] for record in records}) != 1:
        raise ValueError("Adaptive selection accepts candidates for one query at a time")
    return min(
        records,
        key=lambda record: (
            -float(record["combined_score"]),
            int(record["candidate_order"]),
            str(record["candidate_id"]),
        ),
    )
