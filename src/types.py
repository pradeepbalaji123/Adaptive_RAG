"""Serializable domain records with an explicit gold-reference boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RAGExample:
    """Reference-free query record. It deliberately has no answer field."""

    query_id: str
    question: str
    document_id: str
    context: str
    title: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoldRecord:
    """Evaluation-only record, never accepted by adaptive selection code."""

    query_id: str
    answers: tuple[str, ...]
    document_id: str

    def as_dict(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "answers": list(self.answers), "document_id": self.document_id}


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    title: str
    source_example_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    chunk_index: int
    embedding_key: str
    chunk_size: int
    source_example_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    candidate_id: str
    chunks: tuple[Chunk, ...]
    similarity_scores: tuple[float, ...]
    retrieval_latency: float

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"[Evidence {index + 1} | {chunk.chunk_id}]\n{chunk.text}"
            for index, chunk in enumerate(self.chunks)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "candidate_id": self.candidate_id,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in self.chunks],
            "retrieved_chunk_texts": [chunk.text for chunk in self.chunks],
            "similarity_scores": list(self.similarity_scores),
            "retrieval_latency": self.retrieval_latency,
        }


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    latency: float
    input_tokens: int
    output_tokens: int
    seed: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FORBIDDEN_REFERENCE_KEYS = frozenset(
    {"answer", "answers", "gold", "gold_answer", "gold_answers", "reference", "references"}
)


def assert_reference_free_examples(examples: Sequence[RAGExample]) -> None:
    """Fail loudly if gold data crosses the adaptive pipeline boundary."""

    for example in examples:
        if not isinstance(example, RAGExample):
            if isinstance(example, Mapping):
                leaked = FORBIDDEN_REFERENCE_KEYS.intersection(str(key).lower() for key in example)
                if leaked:
                    raise ValueError(f"Gold/reference fields crossed the RAG boundary: {sorted(leaked)}")
            raise TypeError("Adaptive RAG accepts only RAGExample records")
        if not example.question.strip() or not example.context.strip():
            raise ValueError(f"Malformed reference-free record: {example.query_id}")

