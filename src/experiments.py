"""Complete fixed, adaptive, and post-hoc evaluation experiment orchestration."""

from __future__ import annotations

import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # Keep static/core validation usable before Colab installs requirements.
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable

from .cache import CheckpointStore, DiskCache, atomic_write_json, canonical_hash
from .config import CandidateConfig, ExperimentConfig
from .data import (
    create_corpus,
    load_gold_for_evaluation,
    prepare_reference_free_squad_data,
    save_corpus,
)
from .embeddings import SentenceEmbeddingRegistry
from .evaluation import (
    attach_gold_evaluation,
    correlation_analysis,
    csv_safe_records,
    embedding_comparison,
    run_ablations,
    select_oracle_candidates,
    selection_regret_records,
    summarize_systems,
)
from .generation import Phi3Generator
from .plotting import create_research_plots
from .retrieval import FaissIndexManager
from .schemas import validate_result_schema
from .scoring import (
    NLIScorer,
    compute_answer_context_consistency,
    compute_retrieval_confidence,
    estimate_semantic_uncertainty,
    normalize_candidate_signals,
    select_adaptive_candidate,
)
from .types import GenerationResult, RAGExample, RetrievalResult, assert_reference_free_examples
from .utils import (
    cleanup_gpu,
    controlled_seed,
    environment_report,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    set_global_seed,
)


def _generation_from_dict(payload: Mapping[str, Any]) -> GenerationResult:
    return GenerationResult(
        answer=str(payload["answer"]),
        latency=float(payload["latency"]),
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
        seed=None if payload.get("seed") is None else int(payload["seed"]),
    )


class ExperimentRunner:
    """Owns expensive models, indexes, caches, checkpoints, and result export."""

    def __init__(self, config: ExperimentConfig) -> None:
        if config.use_google_drive_cache:
            config = self._mount_google_drive(config)
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.results_dir = Path(config.results_dir)
        self.cache = DiskCache(self.cache_dir / "artifacts")
        self.checkpoints = CheckpointStore(self.cache_dir, config.as_dict())
        self.embeddings = SentenceEmbeddingRegistry(config.device, config.embedding_batch_size)
        self.generator = Phi3Generator(config.generator_model_id, config.device, config.max_new_tokens)
        self.nli = NLIScorer(config.nli_model_id, config.nli_device, config.nli_batch_size)
        self.indexes: FaissIndexManager | None = None
        self.rag_examples: list[RAGExample] = []

    @staticmethod
    def _mount_google_drive(config: ExperimentConfig) -> ExperimentConfig:
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError("use_google_drive_cache=True is supported only inside Google Colab") from exc
        drive.mount("/content/drive")
        base = Path("/content/drive/MyDrive/reference_free_adaptive_rag")
        return config.with_paths(base / "cache", base / "results")

    def prepare(self) -> None:
        import torch

        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        if not torch.cuda.is_available():
            print("WARNING: The full Phi-3 experiment is optimized for a Colab NVIDIA GPU.")
        set_global_seed(self.config.seed)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.rag_examples = prepare_reference_free_squad_data(
            self.config.num_questions,
            self.config.seed,
            self.config.dataset_name,
            self.config.dataset_split,
            self.cache_dir,
        )
        assert_reference_free_examples(self.rag_examples)
        corpus = create_corpus(self.rag_examples)
        save_corpus(corpus, self.cache_dir / "dataset" / "corpus.json")
        self.indexes = FaissIndexManager(corpus, self.embeddings, self.config.chunk_overlap, self.cache_dir)
        self.indexes.build_all(self.config.candidates())
        atomic_write_json(self.results_dir / "config_resolved.json", self.config.as_dict())
        atomic_write_json(self.results_dir / "environment.json", environment_report(self.config.as_dict()))

    def _cached_generation(
        self,
        namespace: str,
        cache_key: dict[str, Any],
        compute: Any,
    ) -> GenerationResult:
        cached = self.cache.get(namespace, cache_key)
        if cached is not None:
            try:
                return _generation_from_dict(cached)
            except (KeyError, TypeError, ValueError):
                pass
        result = compute()
        self.cache.set(namespace, cache_key, result.as_dict())
        return result

    def _deterministic_answer(
        self,
        namespace: str,
        example: RAGExample,
        retrieval: RetrievalResult,
    ) -> GenerationResult:
        key = {
            "model": self.config.generator_model_id,
            "query_id": example.query_id,
            "question": example.question,
            "context": canonical_hash(retrieval.context),
            "do_sample": False,
            "max_new_tokens": self.config.max_new_tokens,
        }
        return self._cached_generation(
            namespace,
            key,
            lambda: self.generator.generate(example.question, retrieval.context, do_sample=False),
        )

    def _sample_answers(
        self,
        example: RAGExample,
        candidate: CandidateConfig,
        retrieval: RetrievalResult,
    ) -> list[GenerationResult]:
        seeds = [
            controlled_seed(self.config.seed, example.query_id, candidate.candidate_id, "uncertainty", index)
            for index in range(self.config.uncertainty_samples)
        ]
        key = {
            "model": self.config.generator_model_id,
            "query_id": example.query_id,
            "candidate": candidate.as_dict(),
            "context": canonical_hash(retrieval.context),
            "seeds": seeds,
            "temperature": self.config.sample_temperature,
            "top_p": self.config.sample_top_p,
            "max_new_tokens": self.config.max_new_tokens,
        }
        cached = self.cache.get("uncertainty_generations", key)
        if cached is not None:
            try:
                results = [_generation_from_dict(item) for item in cached]
                if len(results) == 3:
                    return results
            except (KeyError, TypeError, ValueError):
                pass
        results = self.generator.sample_answers(
            example.question,
            retrieval.context,
            seeds,
            self.config.sample_temperature,
            self.config.sample_top_p,
        )
        self.cache.set("uncertainty_generations", key, [result.as_dict() for result in results])
        return results

    def _semantic_uncertainty(self, answers: Sequence[str]) -> dict[str, Any]:
        key = {
            "model": self.config.uncertainty_embedding_model_id,
            "answers": list(answers),
            "formula": "1-minus-mean-three-pairwise-cosines",
        }
        cached = self.cache.get("semantic_uncertainty", key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        uncertainty, confidence, similarities = estimate_semantic_uncertainty(
            answers,
            lambda texts: self.embeddings.encode(
                "uncertainty_evaluator",
                self.config.uncertainty_embedding_model_id,
                texts,
            ),
        )
        result = {
            "semantic_uncertainty": uncertainty,
            "semantic_confidence": confidence,
            "pairwise_answer_similarities": similarities,
            "semantic_scoring_latency": time.perf_counter() - started,
        }
        self.cache.set("semantic_uncertainty", key, result)
        return result

    def _context_consistency(self, answer: str, retrieval: RetrievalResult) -> dict[str, float]:
        chunk_texts = [chunk.text for chunk in retrieval.chunks]
        key = {
            "model": self.config.nli_model_id,
            "answer": answer,
            "chunks": [canonical_hash(text) for text in chunk_texts],
            "aggregation": "sentence-max-over-chunks-then-mean",
        }
        cached = self.cache.get("nli_scores", key)
        if cached is not None:
            return {name: float(value) for name, value in cached.items()}
        started = time.perf_counter()
        score = compute_answer_context_consistency(answer, chunk_texts, self.nli)
        result = {"context_consistency": score, "nli_latency": time.perf_counter() - started}
        self.cache.set("nli_scores", key, result)
        return result

    def _baseline_one(self, example: RAGExample) -> dict[str, Any]:
        assert self.indexes is not None
        reset_peak_gpu_memory()
        candidate = self.config.baseline_candidate()
        retrieval = self.indexes.retrieve_cached(example.query_id, example.question, candidate, self.cache)
        generation = self._deterministic_answer("baseline_generations", example, retrieval)
        consistency = self._context_consistency(generation.answer, retrieval)
        return {
            "query_id": example.query_id,
            "question": example.question,
            "candidate_id": candidate.candidate_id,
            "configuration_id": candidate.candidate_id,
            "embedding_key": candidate.embedding_key,
            "embedding_model_id": candidate.embedding_model_id,
            "embedding_model": candidate.embedding_model_id,
            "chunk_size": candidate.chunk_size,
            "top_k": candidate.top_k,
            **retrieval.as_dict(),
            "retrieved_document_ids": [chunk.document_id for chunk in retrieval.chunks],
            "retrieved_context": retrieval.context,
            "retrieval_scores": list(retrieval.similarity_scores),
            "generated_answer": generation.answer,
            "context_consistency": consistency["context_consistency"],
            "generation_latency": generation.latency,
            "nli_latency": consistency["nli_latency"],
            "total_latency": retrieval.retrieval_latency + generation.latency + consistency["nli_latency"],
            "llm_generations": 1,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        }

    def run_baseline(self) -> list[dict[str, Any]]:
        records = self.checkpoints.load("baseline")
        processed = {str(record["query_id"]) for record in records}
        remaining = [example for example in self.rag_examples if example.query_id not in processed]
        for count, example in enumerate(tqdm(remaining, desc="Fixed baseline"), start=1):
            records.append(self._baseline_one(example))
            if count % self.config.checkpoint_every == 0:
                self.checkpoints.save("baseline", records)
        self.checkpoints.save("baseline", records)
        return records

    def _candidate_record(self, example: RAGExample, candidate: CandidateConfig) -> dict[str, Any]:
        assert self.indexes is not None
        retrieval = self.indexes.retrieve_cached(example.query_id, example.question, candidate, self.cache)
        sampled = self._sample_answers(example, candidate, retrieval)
        answers = [result.answer for result in sampled]
        uncertainty = self._semantic_uncertainty(answers)
        # The documents specify a singular generated answer for C but three samples for U.
        # We use A1 for C and post-hoc oracle scoring; all A1/A2/A3 contribute to U.
        consistency = self._context_consistency(answers[0], retrieval)
        generation_latency = float(sum(result.latency for result in sampled))
        return {
            "query_id": example.query_id,
            "question": example.question,
            "candidate_id": candidate.candidate_id,
            "configuration_id": candidate.candidate_id,
            "candidate_order": candidate.order,
            "embedding_key": candidate.embedding_key,
            "embedding_model_id": candidate.embedding_model_id,
            "embedding_model": candidate.embedding_model_id,
            "chunk_size": candidate.chunk_size,
            "top_k": candidate.top_k,
            **retrieval.as_dict(),
            "retrieved_document_ids": [chunk.document_id for chunk in retrieval.chunks],
            "retrieved_context": retrieval.context,
            "retrieval_confidence": compute_retrieval_confidence(retrieval.similarity_scores),
            "sample_1": answers[0],
            "sample_2": answers[1],
            "sample_3": answers[2],
            "sampled_answers": answers,
            "generated_answer": answers[0],
            **uncertainty,
            **consistency,
            "candidate_generation_latency": generation_latency,
            "generation_latency": generation_latency,
            "candidate_total_latency": (
                retrieval.retrieval_latency + generation_latency
                + float(uncertainty["semantic_scoring_latency"]) + float(consistency["nli_latency"])
            ),
            "total_latency": (
                retrieval.retrieval_latency + generation_latency
                + float(uncertainty["semantic_scoring_latency"]) + float(consistency["nli_latency"])
            ),
            "llm_generations": 3,
            "input_tokens": sum(result.input_tokens for result in sampled),
            "output_tokens": sum(result.output_tokens for result in sampled),
            "peak_gpu_memory_mb": peak_gpu_memory_mb(),
            "selected": False,
        }

    def _adaptive_one(self, example: RAGExample) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        reset_peak_gpu_memory()
        candidates = [self._candidate_record(example, candidate) for candidate in self.config.candidates()]
        normalized = normalize_candidate_signals(candidates)
        for record in normalized:
            record["R_raw"] = record["retrieval_confidence"]
            record["U_raw"] = record["semantic_uncertainty"]
            record["C_raw"] = record["context_consistency"]
            record["latency"] = record["total_latency"]
        selected = select_adaptive_candidate(normalized)
        for record in normalized:
            record["selected"] = record["candidate_id"] == selected["candidate_id"]

        selected_candidate = next(
            candidate for candidate in self.config.candidates()
            if candidate.candidate_id == selected["candidate_id"]
        )
        assert self.indexes is not None
        selected_retrieval = self.indexes.retrieve_cached(
            example.query_id, example.question, selected_candidate, self.cache
        )
        final_generation = self._deterministic_answer("adaptive_final_generations", example, selected_retrieval)
        final_consistency = self._context_consistency(final_generation.answer, selected_retrieval)
        retrieval_latency = float(sum(record["retrieval_latency"] for record in normalized))
        sampled_generation_latency = float(sum(record["candidate_generation_latency"] for record in normalized))
        scoring_latency = float(sum(
            record["semantic_scoring_latency"] + record["nli_latency"] for record in normalized
        )) + final_consistency["nli_latency"]
        adaptive = {
            "query_id": example.query_id,
            "question": example.question,
            "selected_candidate": selected["candidate_id"],
            "selected_embedding_key": selected["embedding_key"],
            "selected_embedding_model_id": selected["embedding_model_id"],
            "selected_chunk_size": selected["chunk_size"],
            "selected_top_k": selected["top_k"],
            "selected_retrieval_confidence": selected["retrieval_confidence"],
            "selected_semantic_uncertainty": selected["semantic_uncertainty"],
            "selected_semantic_confidence": selected["semantic_confidence"],
            "selected_context_consistency": selected["context_consistency"],
            "selected_combined_score": selected["combined_score"],
            "selected_R": selected["retrieval_confidence"],
            "selected_U": selected["semantic_uncertainty"],
            "selected_C": selected["context_consistency"],
            "selected_J": selected["combined_score"],
            "context_consistency": final_consistency["context_consistency"],
            "retrieved_chunk_ids": selected["retrieved_chunk_ids"],
            "retrieved_chunk_texts": selected["retrieved_chunk_texts"],
            "retrieved_document_ids": selected["retrieved_document_ids"],
            "retrieved_context": selected["retrieved_context"],
            "final_answer": final_generation.answer,
            "retrieval_latency": retrieval_latency,
            "generation_latency": sampled_generation_latency + final_generation.latency,
            "scoring_latency": scoring_latency,
            "final_generation_latency": final_generation.latency,
            "total_latency": retrieval_latency + sampled_generation_latency + final_generation.latency + scoring_latency,
            "llm_generations": len(normalized) * 3 + 1,
            "input_tokens": sum(record["input_tokens"] for record in normalized) + final_generation.input_tokens,
            "output_tokens": sum(record["output_tokens"] for record in normalized) + final_generation.output_tokens,
            "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        }
        adaptive["latency"] = adaptive["total_latency"]
        return normalized, adaptive

    def run_adaptive(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidate_records = self.checkpoints.load("candidates")
        adaptive_records = self.checkpoints.load("adaptive")
        expected = len(self.config.candidates())
        candidate_counts = Counter(str(record["query_id"]) for record in candidate_records)
        complete = {
            str(record["query_id"]) for record in adaptive_records
            if candidate_counts[str(record["query_id"])] == expected
        }
        # Recover cleanly if interruption occurred between the two atomic stage saves.
        adaptive_records = [record for record in adaptive_records if str(record["query_id"]) in complete]
        candidate_records = [record for record in candidate_records if str(record["query_id"]) in complete]
        processed = complete
        remaining = [example for example in self.rag_examples if example.query_id not in processed]
        for count, example in enumerate(tqdm(remaining, desc="Adaptive RAG"), start=1):
            query_candidates, adaptive = self._adaptive_one(example)
            candidate_records.extend(query_candidates)
            adaptive_records.append(adaptive)
            if count % self.config.checkpoint_every == 0:
                self.checkpoints.save("candidates", candidate_records)
                self.checkpoints.save("adaptive", adaptive_records)
        self.checkpoints.save("candidates", candidate_records)
        self.checkpoints.save("adaptive", adaptive_records)
        return candidate_records, adaptive_records

    def _write_csv(self, filename: str, records: Sequence[dict[str, Any]]) -> Path:
        validate_result_schema(filename, records)
        path = self.results_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_safe_records(records)).to_csv(path, index=False)
        return path

    def evaluate_and_export(
        self,
        baseline: Sequence[dict[str, Any]],
        candidates: Sequence[dict[str, Any]],
        adaptive: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        # Gold is first loaded here, after adaptive selections and final answers exist.
        gold = load_gold_for_evaluation(
            self.config.num_questions,
            self.config.seed,
            self.config.dataset_name,
            self.config.dataset_split,
            self.cache_dir,
        )
        baseline_eval = attach_gold_evaluation(baseline, gold, "generated_answer") if baseline else []
        candidate_eval = attach_gold_evaluation(candidates, gold, "generated_answer") if candidates else []
        adaptive_eval = attach_gold_evaluation(adaptive, gold, "final_answer") if adaptive else []
        oracle = select_oracle_candidates(candidate_eval) if self.config.run_oracle and candidate_eval else []
        regrets = selection_regret_records(candidate_eval, oracle) if oracle else []
        correlations = correlation_analysis(candidate_eval) if candidate_eval else []
        ablations = run_ablations(candidate_eval, oracle) if self.config.run_ablations and oracle else []
        embeddings = embedding_comparison(candidate_eval) if candidate_eval else []
        summary = summarize_systems(baseline_eval, adaptive_eval, oracle) if oracle else []
        evaluation_rows = [
            {"system": "Fixed RAG", **record} for record in baseline_eval
        ] + [{"system": "Adaptive RAG", **record} for record in adaptive_eval]
        selected_distribution = dict(Counter(record["selected_candidate"] for record in adaptive))
        denominator = max(len(adaptive), 1)
        selected_percentages = {
            candidate: count / denominator for candidate, count in selected_distribution.items()
        }
        selection_rows = [
            {"candidate_id": candidate, "count": count, "proportion": selected_percentages[candidate]}
            for candidate, count in selected_distribution.items()
        ]

        output_sets = {
            "baseline_results.csv": baseline_eval,
            "candidate_results.csv": candidate_eval,
            "per_query_results.csv": candidate_eval,
            "adaptive_results.csv": adaptive_eval,
            "evaluation_results.csv": evaluation_rows,
            "oracle_results.csv": oracle,
            "selection_regret.csv": regrets,
            "ablation_results.csv": ablations,
            "ablation.csv": ablations,
            "embedding_comparison.csv": embeddings,
            "correlation_results.csv": correlations,
            "selection_distribution.csv": selection_rows,
            "summary.csv": summary,
        }
        paths = [self._write_csv(name, records) for name, records in output_sets.items()]
        regret_summary = {
            "average_regret": float(np.mean([record["selection_regret"] for record in regrets])) if regrets else None,
            "zero_regret_proportion": float(np.mean([record["zero_regret"] for record in regrets])) if regrets else None,
            "oracle_agreement": float(np.mean([record["oracle_agreement"] for record in regrets])) if regrets else None,
        }
        systems_by_name = {record["system"]: record for record in summary}
        fixed = systems_by_name.get("Fixed RAG")
        proposed = systems_by_name.get("Adaptive RAG")
        oracle_summary = systems_by_name.get("Oracle")
        adaptive_vs_fixed = ({
            "delta_exact_match": proposed["exact_match"] - fixed["exact_match"],
            "delta_f1": proposed["f1"] - fixed["f1"],
        } if fixed and proposed else None)
        adaptive_vs_oracle = ({
            "delta_exact_match": proposed["exact_match"] - oracle_summary["exact_match"],
            "delta_f1": proposed["f1"] - oracle_summary["f1"],
        } if proposed and oracle_summary else None)
        summary_payload = {
            "questions_evaluated": len(adaptive_eval or baseline_eval),
            "systems": summary,
            "selection_regret": regret_summary,
            "adaptive_vs_fixed": adaptive_vs_fixed,
            "adaptive_vs_oracle": adaptive_vs_oracle,
            "selected_configuration_distribution": selected_distribution,
            "selected_configuration_percentages": selected_percentages,
            "best_ablation_by_f1": max(ablations, key=lambda row: row["f1"]) if ablations else None,
        }
        atomic_write_json(self.results_dir / "summary_metrics.json", summary_payload)
        paths.append(self.results_dir / "summary_metrics.json")
        plot_paths = create_research_plots(
            summary, candidate_eval, ablations, correlations, self.results_dir / "plots"
        ) if summary and candidate_eval and ablations and correlations else []
        self._print_summary(summary_payload)
        return {
            "paths": [str(path) for path in paths + plot_paths],
            "summary": summary_payload,
            "baseline": baseline_eval,
            "candidates": candidate_eval,
            "adaptive": adaptive_eval,
            "oracle": oracle,
            "ablations": ablations,
        }

    @staticmethod
    def _print_summary(payload: dict[str, Any]) -> None:
        print("\nExperiment summary")
        print(f"Questions evaluated: {payload['questions_evaluated']}")
        for system in payload["systems"]:
            print(
                f"{system['system']}: EM={system['exact_match']:.4f}, F1={system['f1']:.4f}, "
                f"Hit@K={system['retrieval_hit']:.4f}, latency={system['average_total_latency']:.3f}s"
            )
        regret = payload["selection_regret"]
        comparison = payload.get("adaptive_vs_fixed")
        if comparison:
            print(
                f"Adaptive vs Fixed: delta EM={comparison['delta_exact_match']:+.4f}, "
                f"delta F1={comparison['delta_f1']:+.4f}"
            )
        if regret["average_regret"] is not None:
            print(
                f"Adaptive vs Oracle: regret={regret['average_regret']:.4f}, "
                f"agreement={regret['oracle_agreement']:.4f}"
            )
        print("Selected configuration distribution:", payload["selected_configuration_distribution"])
        if payload["best_ablation_by_f1"]:
            print("Best ablation:", payload["best_ablation_by_f1"]["ablation"])

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.prepare()
        baseline = self.run_baseline() if self.config.run_baseline else []
        candidates: list[dict[str, Any]] = []
        adaptive: list[dict[str, Any]] = []
        if self.config.run_adaptive:
            candidates, adaptive = self.run_adaptive()
        outputs = self.evaluate_and_export(baseline, candidates, adaptive)
        outputs["wall_clock_seconds"] = time.perf_counter() - started
        outputs["summary"]["total_experiment_runtime_seconds"] = outputs["wall_clock_seconds"]
        atomic_write_json(self.results_dir / "summary_metrics.json", outputs["summary"])
        cleanup_gpu()
        return outputs


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    return ExperimentRunner(config or ExperimentConfig()).run()
