"""Command-line entry point for Colab, Linux GPU hosts, or local smoke runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig  # noqa: E402
from src.experiments import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference-free adaptive RAG experiments")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--minilm-only", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--results-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.num_questions is not None:
        config.num_questions = args.num_questions
    if args.seed is not None:
        config.seed = args.seed
    if args.minilm_only:
        config.include_bge = False
    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    config.validate()
    run_experiment(config)


if __name__ == "__main__":
    main()

