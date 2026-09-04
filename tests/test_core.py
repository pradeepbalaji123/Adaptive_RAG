from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cache import atomic_write_json
from src.chunking import chunk_documents
from src.config import CandidateConfig, ExperimentConfig
from src.data import _subset_paths, create_corpus, deterministic_subset_indices
from src.evaluation import (
    correlation_analysis,
    evaluate_answer,
    normalize_answer,
    retrieval_hit,
    run_ablations,
    select_oracle_candidates,
    selection_regret_records,
)
from src.generation import build_grounded_messages, validate_generated_answer
from src.experiments import ExperimentRunner
from src.retrieval import FaissIndexManager
from src.plotting import create_research_plots
from src.schemas import validate_result_schema
from src.scoring import (
    compute_answer_context_consistency,
    compute_reference_free_score,
    compute_retrieval_confidence,
    min_max_normalize,
    normalize_candidate_signals,
    select_adaptive_candidate,
    semantic_uncertainty_from_embeddings,
)
from src.types import Chunk, Document, GenerationResult, RAGExample, RetrievalResult, assert_reference_free_examples


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            value = len(self.vocab) + 1000
            self.vocab[token] = value
            self.reverse[value] = token
        return self.vocab[token]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def encode(self, text: str, add_special_tokens: bool, truncation: bool = False) -> list[int]:
        ids = [self._id(token) for token in text.split()]
        return [101, *ids, 102] if add_special_tokens else ids

    def decode(self, ids: list[int], **_: object) -> str:
        return " ".join(self.reverse[value] for value in ids if value in self.reverse)


class FakeEmbeddings:
    def encode(self, *_: object) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FakeIndex:
    ntotal = 3

    def search(self, _: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)[:, :k],
            np.asarray([[0, 1, 2]], dtype=np.int64)[:, :k],
        )


class FakeNLI:
    def entailment_probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.pairs = pairs
        return np.asarray([0.1, 0.8, 0.6, 0.2], dtype=np.float64)


class ConstantFakeNLI:
    def entailment_probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.full(len(pairs), 0.8, dtype=np.float64)


class RunnerFakeEmbeddings:
    def encode(self, key: str, model_id: str, texts: list[str]) -> np.ndarray:
        vectors = []
        for index, _ in enumerate(texts):
            vector = np.asarray([1.0, 0.1 * index], dtype=np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)


class RunnerFakeIndexes:
    def retrieve_cached(self, query_id: str, question: str, candidate: CandidateConfig, cache: object) -> RetrievalResult:
        chunks = tuple(
            Chunk(
                f"{candidate.candidate_id}_chunk_{index}", "d1", f"Paris evidence {index}",
                5, index, candidate.embedding_key, candidate.chunk_size, (query_id,),
            )
            for index in range(candidate.top_k)
        )
        base = candidate.chunk_size / 1000 + candidate.top_k / 100
        scores = tuple(base - index / 1000 for index in range(candidate.top_k))
        return RetrievalResult(query_id, candidate.candidate_id, chunks, scores, 0.01)


class RunnerFakeGenerator:
    def sample_answers(
        self, question: str, context: str, seeds: list[int], temperature: float, top_p: float
    ) -> list[GenerationResult]:
        return [GenerationResult(f"Paris sample {index + 1}.", 0.02, 20, 4, seed) for index, seed in enumerate(seeds)]

    def generate(self, question: str, context: str, **_: object) -> GenerationResult:
        return GenerationResult("Paris.", 0.02, 20, 2, None)


class CoreMethodologyTests(unittest.TestCase):
    def test_candidate_space_and_baseline(self) -> None:
        full = ExperimentConfig()
        self.assertEqual(len(full.candidates()), 8)
        self.assertEqual(full.baseline_candidate().candidate_id, "minilm_c512_k3")
        mini = ExperimentConfig(include_bge=False)
        self.assertEqual(len(mini.candidates()), 4)
        self.assertEqual({(c.chunk_size, c.top_k) for c in mini.candidates()}, {(256, 3), (256, 5), (512, 3), (512, 5)})

    def test_deterministic_subset_and_reference_boundary(self) -> None:
        first = deterministic_subset_indices(100, 20, 42)
        self.assertEqual(first, deterministic_subset_indices(100, 20, 42))
        self.assertNotEqual(first, deterministic_subset_indices(100, 20, 43))
        examples = [RAGExample("q1", "Question?", "d1", "Context text", "Title")]
        assert_reference_free_examples(examples)
        self.assertEqual(create_corpus(examples)[0].source_example_ids, ("q1",))
        with self.assertRaises(ValueError):
            assert_reference_free_examples([{"query_id": "q1", "answer": "leak"}])  # type: ignore[list-item]

    def test_token_aware_chunking_and_overlap(self) -> None:
        tokenizer = FakeTokenizer()
        text = " ".join(f"t{i}" for i in range(14))
        document = Document("d1", text, "", ("q1",))
        chunks = chunk_documents([document], tokenizer, "fake", chunk_size=8, overlap=2)
        self.assertTrue(all(chunk.token_count <= 8 for chunk in chunks))
        first = chunks[0].text.split()
        second = chunks[1].text.split()
        self.assertEqual(first[-2:], second[:2])
        self.assertEqual((chunks[0].document_id, chunks[0].chunk_index), ("d1", 0))

    def test_retrieval_exact_top_k_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = FaissIndexManager([], FakeEmbeddings(), 50, directory)  # type: ignore[arg-type]
            candidate = CandidateConfig("fake_c256_k3", "fake", "fake/model", 256, 3, 0)
            chunks = [Chunk(f"c{i}", "d", f"text {i}", 2, i, "fake", 256, ("q",)) for i in range(3)]
            manager._indexes[("fake", 256)] = FakeIndex()
            manager._chunks[("fake", 256)] = chunks
            result = manager.retrieve("q", "question", candidate)
            self.assertEqual(len(result.chunks), 3)
            self.assertEqual(len(result.similarity_scores), 3)
            self.assertAlmostEqual(result.similarity_scores[0], 0.9, places=6)

    def test_exact_r_u_c_formulas_are_finite(self) -> None:
        self.assertAlmostEqual(compute_retrieval_confidence([0.2, 0.4, 0.6]), 0.4)
        vectors = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float64)
        uncertainty, confidence, similarities = semantic_uncertainty_from_embeddings(vectors)
        self.assertAlmostEqual(confidence, 1 / 3)
        self.assertAlmostEqual(uncertainty, 2 / 3)
        self.assertEqual(len(similarities), 3)
        consistency = compute_answer_context_consistency("One. Two.", ["a", "b"], FakeNLI())  # type: ignore[arg-type]
        self.assertAlmostEqual(consistency, 0.7)
        self.assertTrue(all(math.isfinite(value) for value in [uncertainty, confidence, consistency]))

    def test_normalization_j_and_deterministic_selection(self) -> None:
        self.assertEqual(min_max_normalize([4.0, 4.0]), [0.5, 0.5])
        self.assertEqual(min_max_normalize([2.0, 4.0]), [0.0, 1.0])
        self.assertAlmostEqual(compute_reference_free_score(0.3, 0.6, 0.9), 0.6)
        raw = [
            {"query_id": "q", "candidate_id": "b", "candidate_order": 1, "retrieval_confidence": 1.0, "semantic_confidence": 1.0, "context_consistency": 1.0},
            {"query_id": "q", "candidate_id": "a", "candidate_order": 0, "retrieval_confidence": 1.0, "semantic_confidence": 1.0, "context_consistency": 1.0},
        ]
        normalized = normalize_candidate_signals(raw)
        self.assertTrue(all(record["combined_score"] == 0.5 for record in normalized))
        self.assertEqual(select_adaptive_candidate(normalized)["candidate_id"], "a")
        with self.assertRaises(ValueError):
            select_adaptive_candidate([{**normalized[0], "gold_answers": ["leak"]}])

    def test_prompt_and_generation_validation(self) -> None:
        messages = build_grounded_messages("Who?", "Evidence")
        self.assertIn("untrusted", messages[0]["content"])
        self.assertIn("<retrieved_evidence>", messages[1]["content"])
        self.assertEqual(validate_generated_answer(" Paris "), "Paris")
        with self.assertRaises(RuntimeError):
            validate_generated_answer("  ")

    def test_squad_metrics_and_retrieval_hit(self) -> None:
        self.assertEqual(normalize_answer("The, Eiffel Tower!"), "eiffel tower")
        metrics = evaluate_answer("Eiffel Tower", ["the Eiffel Tower", "another answer"])
        self.assertEqual(metrics, {"exact_match": 1.0, "f1": 1.0})
        self.assertEqual(retrieval_hit(["It is located at the Eiffel Tower."], ["Eiffel Tower"]), 1.0)

    def test_required_candidate_export_schema(self) -> None:
        record = {key: None for key in {
            "query_id", "candidate_id", "embedding_model_id", "chunk_size", "top_k",
            "retrieval_confidence", "semantic_uncertainty", "semantic_confidence", "context_consistency",
            "combined_score", "generated_answer", "selected", "gold_answers", "exact_match", "f1", "retrieval_hit",
            "R_raw", "U_raw", "C_raw", "R_normalized", "U_normalized", "C_normalized", "J",
        }}
        validate_result_schema("candidate_results.csv", [record])
        with self.assertRaises(ValueError):
            validate_result_schema("candidate_results.csv", [{"query_id": "q"}])

    def test_inexpensive_runner_integration_without_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                num_questions=1,
                include_bge=False,
                cache_dir=str(Path(directory) / "cache"),
                results_dir=str(Path(directory) / "results"),
            )
            runner = ExperimentRunner(config)
            runner.embeddings = RunnerFakeEmbeddings()  # type: ignore[assignment]
            runner.generator = RunnerFakeGenerator()  # type: ignore[assignment]
            runner.nli = ConstantFakeNLI()  # type: ignore[assignment]
            runner.indexes = RunnerFakeIndexes()  # type: ignore[assignment]
            example = RAGExample("q1", "Where?", "d1", "Paris is in France.", "")
            self.assertFalse(hasattr(runner, "gold"))
            baseline = runner._baseline_one(example)
            candidates, adaptive = runner._adaptive_one(example)
            self.assertEqual(baseline["generated_answer"], "Paris.")
            self.assertEqual(len(candidates), 4)
            self.assertEqual(sum(bool(record["selected"]) for record in candidates), 1)
            self.assertEqual(adaptive["final_answer"], "Paris.")
            self.assertEqual(adaptive["llm_generations"], 13)
            self.assertFalse(any("gold" in key or "reference" in key for record in candidates for key in record))
            _, gold_path = _subset_paths(
                config.num_questions, config.seed, config.dataset_name,
                config.dataset_split, config.cache_dir,
            )
            atomic_write_json(gold_path, [{
                "query_id": "q1", "answers": ["Paris"], "document_id": "d1",
            }])
            os.environ["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
            outputs = runner.evaluate_and_export([baseline], candidates, [adaptive])
            self.assertEqual(outputs["summary"]["questions_evaluated"], 1)
            for filename in [
                "baseline_results.csv", "candidate_results.csv", "per_query_results.csv",
                "adaptive_results.csv", "oracle_results.csv", "selection_regret.csv",
                "ablation_results.csv", "ablation.csv", "embedding_comparison.csv",
                "correlation_results.csv", "selection_distribution.csv", "summary.csv",
                "summary_metrics.json",
            ]:
                self.assertTrue((Path(config.results_dir) / filename).exists(), filename)
            validate_result_schema(
                "baseline_results.csv",
                [{**baseline, "exact_match": 1.0, "f1": 1.0, "retrieval_hit": 1.0}],
            )
            validate_result_schema(
                "adaptive_results.csv",
                [{**adaptive, "exact_match": 1.0, "f1": 1.0, "retrieval_hit": 1.0}],
            )

    def test_posthoc_analysis_and_plot_exports(self) -> None:
        candidates = []
        for query_index in range(2):
            for candidate_order in range(2):
                value = (query_index + candidate_order) / 2
                candidates.append({
                    "query_id": f"q{query_index}",
                    "candidate_id": f"c{candidate_order}",
                    "candidate_order": candidate_order,
                    "embedding_key": "minilm",
                    "embedding_model_id": "model",
                    "retrieval_confidence": value,
                    "semantic_uncertainty": 1 - value,
                    "semantic_confidence": value,
                    "context_consistency": value,
                    "combined_score": value,
                    "retrieval_confidence_normalized": value,
                    "semantic_confidence_normalized": value,
                    "context_consistency_normalized": value,
                    "exact_match": float(candidate_order == 1),
                    "f1": float(candidate_order == 1),
                    "retrieval_hit": float(candidate_order == 1),
                    "selected": candidate_order == 0,
                })
        oracle = select_oracle_candidates(candidates)
        regrets = selection_regret_records(candidates, oracle)
        correlations = correlation_analysis(candidates)
        ablations = run_ablations(candidates, oracle)
        self.assertEqual(len(oracle), 2)
        self.assertEqual(len(regrets), 2)
        self.assertEqual(len(correlations), 10)
        self.assertEqual(len(ablations), 7)
        summary = [
            {
                "system": system,
                "exact_match": 0.5,
                "f1": 0.6,
                "retrieval_hit": 0.7,
                "average_retrieval_latency": 0.01,
                "average_generation_latency": 0.02,
                "average_total_latency": 0.03,
            }
            for system in ["Fixed RAG", "Adaptive RAG", "Oracle"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            os.environ["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
            paths = create_research_plots(summary, candidates, ablations, correlations, Path(directory) / "plots")
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
