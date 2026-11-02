from __future__ import annotations

import unittest
import numpy as np

from src.cascade import (
    CascadedConfig,
    CascadedAdaptivePipeline,
    compute_lexical_consistency,
)
from src.config import CandidateConfig
from src.types import Chunk, GenerationResult, RetrievalResult


class FakeIndexManager:
    def __init__(self, high_r: bool = True) -> None:
        self.high_r = high_r

    def retrieve(self, query_id: str, question: str, candidate: CandidateConfig) -> RetrievalResult:
        if self.high_r and candidate.candidate_id == "bge_small_c256_k3":
            scores = (0.95, 0.92, 0.90)
        elif candidate.candidate_id == "minilm_c512_k3":
            scores = (0.75, 0.72, 0.70)
        else:
            scores = (0.70, 0.68, 0.65)
        chunks = tuple(
            Chunk(
                chunk_id=f"c_{i}",
                document_id="doc_1",
                text=f"Chunk evidence {i} with factual content about Paris and France.",
                token_count=10,
                chunk_index=i,
                embedding_key=candidate.embedding_key,
                chunk_size=candidate.chunk_size,
                source_example_ids=("ex_1",),
            )
            for i in range(candidate.top_k)
        )
        return RetrievalResult(
            query_id=query_id,
            candidate_id=candidate.candidate_id,
            chunks=chunks,
            similarity_scores=scores,
            retrieval_latency=0.005,
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, context: str, do_sample: bool = False, **_: object) -> GenerationResult:
        self.calls += 1
        return GenerationResult("Paris is the capital of France.", 0.01, 10, 5, 42)

    def sample_answers(
        self,
        question: str,
        context: str,
        seeds: list[int],
        temperature: float = 0.7,
        top_p: float = 0.9,
        *args: object,
        **kwargs: object,
    ) -> list[GenerationResult]:
        self.calls += len(seeds)
        return [
            GenerationResult(f"Paris is the capital of France (sample {s}).", 0.01, 10, 5, s)
            for s in seeds
        ]


class FakeEmbeddings:
    def encode(self, *_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32)


class TestCascadedRAG(unittest.TestCase):
    def test_compute_lexical_consistency(self) -> None:
        chunks = ["Paris is the capital and largest city of France.", "France is located in Europe."]
        # Perfect containment
        score_high = compute_lexical_consistency("Paris is the capital of France.", chunks)
        self.assertGreaterEqual(score_high, 0.8)

        # Zero containment
        score_zero = compute_lexical_consistency("Tokyo is in Japan.", chunks)
        self.assertEqual(score_zero, 0.0)

        # Empty handling
        self.assertEqual(compute_lexical_consistency("", chunks), 0.0)
        self.assertEqual(compute_lexical_consistency("Something", []), 0.0)

        # Explicit abstention phrases must receive 0.0 grounding
        abstention = "The provided evidence does not contain specific information about when PPO was developed."
        self.assertEqual(compute_lexical_consistency(abstention, chunks), 0.0)

    def test_stratified_family_retention(self) -> None:
        """Verify that Stage 1 preserves retriever diversity across embedding families."""
        candidates = [
            CandidateConfig("bge_c256_k3", "bge_small", "m1", 256, 3, 0),
            CandidateConfig("bge_c512_k3", "bge_small", "m1", 512, 3, 1),
            CandidateConfig("minilm_c256_k3", "minilm", "m2", 256, 3, 2),
            CandidateConfig("minilm_c512_k3", "minilm", "m2", 512, 3, 3),
        ]
        # Fake index where BGE has raw scores ~0.68 and MiniLM ~0.43
        class IncommensurableIndexManager:
            def retrieve(self, query_id: str, question: str, candidate: CandidateConfig) -> RetrievalResult:
                score = 0.68 if candidate.embedding_key == "bge_small" else 0.43
                chunks = tuple(
                    Chunk(
                        chunk_id=f"c_{i}",
                        document_id="doc_1",
                        text="Sample evidence",
                        token_count=10,
                        chunk_index=i,
                        embedding_key=candidate.embedding_key,
                        chunk_size=candidate.chunk_size,
                        source_example_ids=("ex_1",),
                    )
                    for i in range(candidate.top_k)
                )
                return RetrievalResult(
                    query_id=query_id,
                    candidate_id=candidate.candidate_id,
                    chunks=chunks,
                    similarity_scores=(score,),
                    retrieval_latency=0.001,
                )

        pipeline = CascadedAdaptivePipeline(
            indexes=IncommensurableIndexManager(),
            generator=FakeGenerator(),
            embeddings=FakeEmbeddings(),
            candidates=candidates,
            cascade_config=CascadedConfig(
                early_exit_threshold=0.85,
                early_exit_margin=0.12,
                top_candidates_to_keep=2,
            ),
        )
        res = pipeline.run_query("When was PPO developed?", mode="cascaded")
        retained_families = {
            s.embedding_key for s in res.all_candidates if s.retained_for_stage_2
        }
        # Both embedding families must be retained in Stage 2!
        self.assertIn("bge_small", retained_families)
        self.assertIn("minilm", retained_families)

    def test_early_exit_triggers_on_high_confidence(self) -> None:
        candidates = [
            CandidateConfig("bge_small_c256_k3", "bge_small", "model", 256, 3, 0),
            CandidateConfig("minilm_c512_k3", "minilm", "model", 512, 3, 1),
        ]
        pipeline = CascadedAdaptivePipeline(
            indexes=FakeIndexManager(high_r=True),
            generator=FakeGenerator(),
            embeddings=FakeEmbeddings(),
            candidates=candidates,
            cascade_config=CascadedConfig(early_exit_threshold=0.85),
        )
        result = pipeline.run_query("What is the capital of France?", mode="cascaded")
        self.assertTrue(result.is_early_exit)
        self.assertEqual(result.selected_candidate_id, "bge_small_c256_k3")
        self.assertEqual(result.llm_generations_count, 1)  # Only 1 generation!

    def test_pruned_stage_two_when_ambiguous(self) -> None:
        candidates = [
            CandidateConfig("cand_1", "bge_small", "model", 256, 3, 0),
            CandidateConfig("cand_2", "minilm", "model", 512, 3, 1),
            CandidateConfig("cand_3", "minilm", "model", 256, 3, 2),
        ]
        # Equal/close scores below threshold
        pipeline = CascadedAdaptivePipeline(
            indexes=FakeIndexManager(high_r=False),
            generator=FakeGenerator(),
            embeddings=FakeEmbeddings(),
            candidates=candidates,
            cascade_config=CascadedConfig(
                early_exit_threshold=0.99,
                early_exit_margin=0.50,
                min_relevance_threshold=0.0,
                top_candidates_to_keep=2,
                reuse_sample_1_as_final=True,
                progressive_stage2=False,
            ),
        )
        result = pipeline.run_query("What is the capital of France?", mode="cascaded")
        self.assertFalse(result.is_early_exit)
        self.assertEqual(result.retained_candidates_count, 2)
        # 2 retained * 3 samples + reused sample 1 = 6 calls
        self.assertEqual(result.llm_generations_count, 6)
        self.assertIn("capital of France", result.answer)

    def test_low_relevance_query_fast_shortcut(self) -> None:
        """Queries like '2+2 =' with low relevance across all docs should exit with 1 call."""
        class LowRelevanceIndexManager:
            def retrieve(self, query_id: str, question: str, candidate: CandidateConfig) -> RetrievalResult:
                chunks = tuple(
                    Chunk(
                        chunk_id=f"c_{i}",
                        document_id="doc_1",
                        text="Unrelated corpus text",
                        token_count=10,
                        chunk_index=i,
                        embedding_key=candidate.embedding_key,
                        chunk_size=candidate.chunk_size,
                        source_example_ids=("ex_1",),
                    )
                    for i in range(candidate.top_k)
                )
                return RetrievalResult(
                    query_id=query_id,
                    candidate_id=candidate.candidate_id,
                    chunks=chunks,
                    similarity_scores=(0.18, 0.15),
                    retrieval_latency=0.001,
                )

        candidates = [
            CandidateConfig("cand_1", "minilm", "m1", 256, 3, 0),
            CandidateConfig("cand_2", "bge_small", "m2", 256, 3, 1),
        ]
        pipeline = CascadedAdaptivePipeline(
            indexes=LowRelevanceIndexManager(),
            generator=FakeGenerator(),
            embeddings=FakeEmbeddings(),
            candidates=candidates,
            cascade_config=CascadedConfig(min_relevance_threshold=0.30),
        )
        result = pipeline.run_query("2+2 =", mode="cascaded")
        self.assertTrue(result.is_early_exit)
        self.assertEqual(result.llm_generations_count, 1)


if __name__ == "__main__":
    unittest.main()
