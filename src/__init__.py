"""Reference-free adaptive retrieval tuning research implementation."""

# Keep the package root lightweight. The full runner is imported explicitly from
# ``src.experiments`` after Colab installs the research dependencies.
from .config import CandidateConfig, ExperimentConfig

__all__ = ["CandidateConfig", "ExperimentConfig"]
