"""Required result columns and validation before research exports."""

from __future__ import annotations

from typing import Any, Sequence


REQUIRED_COLUMNS: dict[str, set[str]] = {
    "baseline_results.csv": {
        "query_id", "question", "retrieved_context", "similarity_scores", "retrieval_scores", "generated_answer",
        "retrieval_latency", "generation_latency", "total_latency", "exact_match", "f1", "retrieval_hit",
    },
    "candidate_results.csv": {
        "query_id", "candidate_id", "embedding_model_id", "chunk_size", "top_k",
        "retrieval_confidence", "semantic_uncertainty", "semantic_confidence", "context_consistency",
        "combined_score", "generated_answer", "selected", "gold_answers", "exact_match", "f1", "retrieval_hit",
        "R_raw", "U_raw", "C_raw", "R_normalized", "U_normalized", "C_normalized", "J",
    },
    "adaptive_results.csv": {
        "query_id", "selected_candidate", "selected_chunk_size", "selected_top_k",
        "selected_retrieval_confidence", "selected_semantic_uncertainty",
        "selected_context_consistency", "selected_combined_score", "final_answer",
        "selected_R", "selected_U", "selected_C", "selected_J",
        "total_latency", "exact_match", "f1", "retrieval_hit",
    },
}


def validate_result_schema(filename: str, records: Sequence[dict[str, Any]]) -> None:
    required = REQUIRED_COLUMNS.get(filename)
    if not required or not records:
        return
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{filename} record {index} is missing columns: {sorted(missing)}")
