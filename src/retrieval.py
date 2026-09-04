"""FAISS inner-product indexes over normalized vectors (cosine retrieval)."""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cache import DiskCache, atomic_write_json, canonical_hash
from .chunking import chunk_documents
from .config import CandidateConfig
from .embeddings import SentenceEmbeddingRegistry
from .types import Chunk, Document, RetrievalResult


class FaissIndexManager:
    def __init__(
        self,
        corpus: Sequence[Document],
        embedding_registry: SentenceEmbeddingRegistry,
        overlap: int,
        cache_dir: str | Path,
    ) -> None:
        self.corpus = list(corpus)
        self.embeddings = embedding_registry
        self.overlap = overlap
        self.cache_dir = Path(cache_dir) / "indexes"
        self._indexes: dict[tuple[str, int], Any] = {}
        self._chunks: dict[tuple[str, int], list[Chunk]] = {}
        self._fingerprints: dict[tuple[str, int], str] = {}

    def _fingerprint(self, candidate: CandidateConfig) -> str:
        return canonical_hash({
            "model": candidate.embedding_model_id,
            "embedding_key": candidate.embedding_key,
            "chunk_size": candidate.chunk_size,
            "overlap": self.overlap,
            "documents": [(doc.document_id, canonical_hash(doc.text)) for doc in self.corpus],
        })

    @staticmethod
    def _chunk_from_dict(payload: dict[str, Any]) -> Chunk:
        allowed = {field.name for field in fields(Chunk)}
        values = {key: value for key, value in payload.items() if key in allowed}
        values["source_example_ids"] = tuple(values["source_example_ids"])
        return Chunk(**values)

    def build(self, candidate: CandidateConfig) -> None:
        key = (candidate.embedding_key, candidate.chunk_size)
        if key in self._indexes:
            return
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install faiss-cpu before building retrieval indexes") from exc

        fingerprint = self._fingerprint(candidate)
        directory = self.cache_dir / f"{candidate.embedding_key}_c{candidate.chunk_size}_{fingerprint}"
        chunks_path = directory / "chunks.json"
        vectors_path = directory / "embeddings.npy"
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"
        if all(path.exists() for path in (chunks_path, vectors_path, index_path, meta_path)):
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata.get("fingerprint") == fingerprint:
                    chunks = [self._chunk_from_dict(item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
                    vectors = np.load(vectors_path, mmap_mode="r")
                    index = faiss.read_index(str(index_path))
                    if len(chunks) == index.ntotal == len(vectors):
                        self._chunks[key], self._indexes[key] = chunks, index
                        self._fingerprints[key] = fingerprint
                        return
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass

        tokenizer = self.embeddings.tokenizer(candidate.embedding_key, candidate.embedding_model_id)
        chunks = chunk_documents(
            self.corpus, tokenizer, candidate.embedding_key, candidate.chunk_size, self.overlap
        )
        vectors = self.embeddings.encode(
            candidate.embedding_key, candidate.embedding_model_id, [chunk.text for chunk in chunks]
        )
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        if index.ntotal != len(chunks):
            raise RuntimeError("FAISS index size does not match chunk metadata")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, vectors)
        faiss.write_index(index, str(index_path))
        atomic_write_json(chunks_path, [chunk.as_dict() for chunk in chunks])
        atomic_write_json(meta_path, {"fingerprint": fingerprint, "count": len(chunks), "dimension": vectors.shape[1]})
        self._chunks[key], self._indexes[key] = chunks, index
        self._fingerprints[key] = fingerprint

    def build_all(self, candidates: Sequence[CandidateConfig]) -> None:
        seen: set[tuple[str, int]] = set()
        for candidate in candidates:
            key = (candidate.embedding_key, candidate.chunk_size)
            if key not in seen:
                self.build(candidate)
                seen.add(key)

    def retrieve(self, query_id: str, question: str, candidate: CandidateConfig) -> RetrievalResult:
        if not question.strip():
            raise ValueError("Cannot retrieve for an empty question")
        self.build(candidate)
        key = (candidate.embedding_key, candidate.chunk_size)
        index, chunks = self._indexes[key], self._chunks[key]
        requested = min(candidate.top_k, len(chunks))
        if requested == 0:
            raise RuntimeError("Retrieval index is empty")
        started = time.perf_counter()
        query_vector = self.embeddings.encode(candidate.embedding_key, candidate.embedding_model_id, [question])
        scores, positions = index.search(query_vector, requested)
        latency = time.perf_counter() - started
        valid = [(float(score), int(position)) for score, position in zip(scores[0], positions[0]) if position >= 0]
        if len(chunks) >= candidate.top_k and len(valid) != candidate.top_k:
            raise RuntimeError(f"Expected exactly {candidate.top_k} retrieval results, got {len(valid)}")
        if not valid:
            raise RuntimeError("FAISS returned no valid retrieval results")
        return RetrievalResult(
            query_id=query_id,
            candidate_id=candidate.candidate_id,
            chunks=tuple(chunks[position] for _, position in valid),
            similarity_scores=tuple(score for score, _ in valid),
            retrieval_latency=latency,
        )

    def retrieve_cached(
        self,
        query_id: str,
        question: str,
        candidate: CandidateConfig,
        cache: DiskCache,
    ) -> RetrievalResult:
        self.build(candidate)
        key = (candidate.embedding_key, candidate.chunk_size)
        cache_key = {
            "query_id": query_id,
            "question": question,
            "candidate": candidate.as_dict(),
            "index": self._fingerprints[key],
        }
        cached = cache.get("retrieval", cache_key)
        if cached is not None:
            by_id = {chunk.chunk_id: chunk for chunk in self._chunks[key]}
            try:
                chunks = tuple(by_id[value] for value in cached["retrieved_chunk_ids"])
                return RetrievalResult(
                    query_id, candidate.candidate_id, chunks,
                    tuple(float(value) for value in cached["similarity_scores"]),
                    float(cached["retrieval_latency"]),
                )
            except (KeyError, TypeError, ValueError):
                pass
        result = self.retrieve(query_id, question, candidate)
        cache.set("retrieval", cache_key, result.as_dict())
        return result

