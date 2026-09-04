"""Post-selection SQuAD evaluation, oracle, regret, correlation, and ablations."""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .types import GoldRecord


def normalize_answer(text: str) -> str:
    """Official SQuAD normalization: lowercase, punctuation/articles removal, whitespace."""

    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str, references: Sequence[str]) -> dict[str, float]:
    if not references:
        raise ValueError("At least one SQuAD reference is required for evaluation")
    return {
        "exact_match": max(exact_match_score(prediction, reference) for reference in references),
        "f1": max(token_f1_score(prediction, reference) for reference in references),
    }


def retrieval_hit(retrieved_texts: Sequence[str], references: Sequence[str]) -> float:
    contexts = [normalize_answer(text) for text in retrieved_texts]
    valid_references = [normalize_answer(reference) for reference in references if normalize_answer(reference)]
    return float(any(reference in context for reference in valid_references for context in contexts))


def attach_gold_evaluation(
    records: Sequence[dict[str, Any]],
    gold: Mapping[str, GoldRecord],
    answer_field: str,
) -> list[dict[str, Any]]:
    """This is the first and only stage where hidden references join predictions."""

    evaluated: list[dict[str, Any]] = []
    for record in records:
        query_id = str(record["query_id"])
        if query_id not in gold:
            raise KeyError(f"Missing gold record for query {query_id}")
        references = gold[query_id].answers
        item = deepcopy(record)
        item["gold_answers"] = list(references)
        item["gold_answer"] = references[0]
        item["gold_answer"] = references[0]
        item.update(evaluate_answer(str(record[answer_field]), references))
        texts = record.get("retrieved_chunk_texts", [])
        item["retrieval_hit"] = retrieval_hit(texts, references)
        item["relevant_document_hit"] = float(
            gold[query_id].document_id in set(record.get("retrieved_document_ids", []))
        )
        evaluated.append(item)
    return evaluated


def select_oracle_candidates(candidate_evaluations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Post-hoc reference-based oracle; never called by adaptive inference."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_evaluations:
        grouped[str(record["query_id"])].append(record)
    oracle: list[dict[str, Any]] = []
    for query_id, records in grouped.items():
        best = min(
            records,
            key=lambda record: (
                -float(record["f1"]),
                -float(record["exact_match"]),
                int(record["candidate_order"]),
            ),
        )
        item = deepcopy(best)
        item["oracle_selected"] = True
        oracle.append(item)
    return oracle


def selection_regret_records(
    candidate_evaluations: Sequence[dict[str, Any]],
    oracle_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {str(record["query_id"]): record for record in candidate_evaluations if record.get("selected")}
    oracle = {str(record["query_id"]): record for record in oracle_records}
    results: list[dict[str, Any]] = []
    for query_id in sorted(oracle):
        proposed = selected[query_id]
        best = oracle[query_id]
        regret = float(best["f1"]) - float(proposed["f1"])
        results.append({
            "query_id": query_id,
            "adaptive_candidate_id": proposed["candidate_id"],
            "oracle_candidate_id": best["candidate_id"],
            "adaptive_candidate_f1": proposed["f1"],
            "oracle_f1": best["f1"],
            "selection_regret": regret,
            "zero_regret": float(abs(regret) <= 1e-12),
            "oracle_agreement": float(proposed["candidate_id"] == best["candidate_id"]),
        })
    return results


def correlation_analysis(candidate_evaluations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(candidate_evaluations)
    signals = [
        "retrieval_confidence",
        "semantic_uncertainty",
        "semantic_confidence",
        "context_consistency",
        "combined_score",
    ]
    targets = ["exact_match", "f1"]
    rows: list[dict[str, Any]] = []
    for signal in signals:
        for target in targets:
            coefficient = float(frame[[signal, target]].corr(method="spearman").iloc[0, 1])
            rows.append({
                "signal": signal,
                "target": target,
                "spearman": coefficient if np.isfinite(coefficient) else None,
                "n": int(frame[[signal, target]].dropna().shape[0]),
            })
    return rows


ABLATION_SIGNALS: dict[str, tuple[str, ...]] = {
    "R_only": ("retrieval_confidence_normalized",),
    "1-U_only": ("semantic_confidence_normalized",),
    "C_only": ("context_consistency_normalized",),
    "R_plus_1-U": ("retrieval_confidence_normalized", "semantic_confidence_normalized"),
    "R_plus_C": ("retrieval_confidence_normalized", "context_consistency_normalized"),
    "1-U_plus_C": ("semantic_confidence_normalized", "context_consistency_normalized"),
    "R_plus_1-U_plus_C": (
        "retrieval_confidence_normalized",
        "semantic_confidence_normalized",
        "context_consistency_normalized",
    ),
}


def run_ablations(
    candidate_evaluations: Sequence[dict[str, Any]],
    oracle_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_evaluations:
        grouped[str(record["query_id"])].append(record)
    oracle_by_query = {str(record["query_id"]): record for record in oracle_records}
    rows: list[dict[str, Any]] = []
    for ablation_name, fields in ABLATION_SIGNALS.items():
        selections: list[dict[str, Any]] = []
        for query_id, candidates in grouped.items():
            best = min(candidates, key=lambda record: (
                -float(np.mean([record[field] for field in fields])),
                int(record["candidate_order"]),
            ))
            selections.append(best)
        distribution = Counter(record["candidate_id"] for record in selections)
        regrets = [
            float(oracle_by_query[str(record["query_id"])]["f1"]) - float(record["f1"])
            for record in selections
        ]
        rows.append({
            "ablation": ablation_name,
            "signals": list(fields),
            "exact_match": float(np.mean([record["exact_match"] for record in selections])),
            "f1": float(np.mean([record["f1"] for record in selections])),
            "retrieval_hit": float(np.mean([record["retrieval_hit"] for record in selections])),
            "average_regret": float(np.mean(regrets)),
            "zero_regret_proportion": float(np.mean([abs(value) <= 1e-12 for value in regrets])),
            "oracle_agreement": float(np.mean([
                record["candidate_id"] == oracle_by_query[str(record["query_id"])]["candidate_id"]
                for record in selections
            ])),
            "selection_distribution": dict(distribution),
        })
    return rows


def _system_row(system: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float:
        values = [float(record.get(field, 0.0)) for record in records]
        return float(np.mean(values)) if values else float("nan")
    return {
        "system": system,
        "questions": len(records),
        "exact_match": mean("exact_match"),
        "f1": mean("f1"),
        "retrieval_hit": mean("retrieval_hit"),
        "answer_context_consistency": mean("context_consistency"),
        "average_retrieval_latency": mean("retrieval_latency"),
        "average_generation_latency": mean("generation_latency"),
        "average_total_latency": mean("total_latency"),
        "average_llm_generations": mean("llm_generations"),
        "average_input_tokens": mean("input_tokens"),
        "average_output_tokens": mean("output_tokens"),
        "average_peak_gpu_memory_mb": mean("peak_gpu_memory_mb"),
    }


def summarize_systems(
    baseline_evaluations: Sequence[dict[str, Any]],
    adaptive_evaluations: Sequence[dict[str, Any]],
    oracle_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    oracle_augmented = []
    for record in oracle_records:
        item = deepcopy(record)
        item.setdefault("generation_latency", item.get("candidate_generation_latency", 0.0))
        item.setdefault("total_latency", item.get("candidate_total_latency", 0.0))
        item.setdefault("llm_generations", 1)
        oracle_augmented.append(item)
    return [
        _system_row("Fixed RAG", baseline_evaluations),
        _system_row("Adaptive RAG", adaptive_evaluations),
        _system_row("Oracle", oracle_augmented),
    ]


def embedding_comparison(candidate_evaluations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(candidate_evaluations)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby("embedding_key", sort=False):
        rows.append({
            "embedding_key": key,
            "embedding_model_id": group["embedding_model_id"].iloc[0],
            "candidate_rows": int(len(group)),
            "mean_retrieval_confidence": float(group["retrieval_confidence"].mean()),
            "mean_retrieval_hit": float(group["retrieval_hit"].mean()),
            "mean_exact_match": float(group["exact_match"].mean()),
            "mean_f1": float(group["f1"].mean()),
            "mean_retrieval_latency": float(group["retrieval_latency"].mean()),
            "selected_count": int(group["selected"].sum()),
        })
    return rows


def csv_safe_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for record in records:
        item = {}
        for key, value in record.items():
            item[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
        safe.append(item)
    return safe
