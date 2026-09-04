"""Experiment configuration and methodology-defined candidate construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    embedding_key: str
    embedding_model_id: str
    chunk_size: int
    top_k: int
    order: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    seed: int = 42
    num_questions: int = 20
    dataset_name: str = "rajpurkar/squad"
    dataset_split: str = "validation"
    generator_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    nli_model_id: str = "cross-encoder/nli-deberta-v3-base"
    uncertainty_embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EMBEDDING_MODELS))
    include_bge: bool = True
    chunk_sizes: tuple[int, ...] = (256, 512)
    top_k_values: tuple[int, ...] = (3, 5)
    chunk_overlap: int = 50
    baseline_embedding: str = "minilm"
    baseline_chunk_size: int = 512
    baseline_top_k: int = 3
    uncertainty_samples: int = 3
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9
    max_new_tokens: int = 96
    embedding_batch_size: int = 64
    nli_batch_size: int = 16
    checkpoint_every: int = 10
    run_baseline: bool = True
    run_adaptive: bool = True
    run_oracle: bool = True
    run_ablations: bool = True
    use_google_drive_cache: bool = False
    cache_dir: str = "cache"
    results_dir: str = "results"
    device: str = "auto"
    nli_device: str = "auto"

    def __post_init__(self) -> None:
        self.chunk_sizes = tuple(int(x) for x in self.chunk_sizes)
        self.top_k_values = tuple(int(x) for x in self.top_k_values)
        self.validate()

    def validate(self) -> None:
        if self.num_questions <= 0:
            raise ValueError("num_questions must be positive")
        if self.uncertainty_samples != 3:
            raise ValueError("The methodology requires exactly three uncertainty samples")
        if self.chunk_overlap < 0 or any(size <= self.chunk_overlap for size in self.chunk_sizes):
            raise ValueError("Every chunk size must exceed the non-negative chunk overlap")
        if any(k <= 0 for k in self.top_k_values):
            raise ValueError("top_k values must be positive")
        if self.baseline_embedding not in self.embedding_models:
            raise ValueError("baseline_embedding is not registered")
        if self.baseline_chunk_size not in self.chunk_sizes or self.baseline_top_k not in self.top_k_values:
            raise ValueError("Fixed baseline configuration must be in the candidate search space")
        if self.sample_temperature <= 0 or not 0 < self.sample_top_p <= 1:
            raise ValueError("Invalid stochastic generation settings")

    def candidates(self) -> list[CandidateConfig]:
        keys = ["minilm"] + (["bge_small"] if self.include_bge else [])
        missing = [key for key in keys if key not in self.embedding_models]
        if missing:
            raise ValueError(f"Missing embedding model registrations: {missing}")
        candidates: list[CandidateConfig] = []
        for order, (key, size, top_k) in enumerate(
            (key, size, top_k)
            for key in keys
            for size in self.chunk_sizes
            for top_k in self.top_k_values
        ):
            candidates.append(CandidateConfig(
                candidate_id=f"{key}_c{size}_k{top_k}",
                embedding_key=key,
                embedding_model_id=self.embedding_models[key],
                chunk_size=size,
                top_k=top_k,
                order=order,
            ))
        return candidates

    def baseline_candidate(self) -> CandidateConfig:
        for candidate in self.candidates():
            if (candidate.embedding_key, candidate.chunk_size, candidate.top_k) == (
                self.baseline_embedding, self.baseline_chunk_size, self.baseline_top_k
            ):
                return candidate
        raise ValueError("Fixed baseline configuration is unavailable")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_paths(self, cache_dir: str | Path, results_dir: str | Path) -> "ExperimentConfig":
        return replace(self, cache_dir=str(cache_dir), results_dir=str(results_dir))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read experiment configuration") from exc
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(**(yaml.safe_load(stream) or {}))

