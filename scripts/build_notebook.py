"""Build the self-contained Colab notebook from the tested source modules."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "reference_free_adaptive_rag_colab.ipynb"
SOURCE_FILES = [
    "src/__init__.py",
    "src/cache.py",
    "src/chunking.py",
    "src/config.py",
    "src/data.py",
    "src/embeddings.py",
    "src/evaluation.py",
    "src/experiments.py",
    "src/generation.py",
    "src/plotting.py",
    "src/retrieval.py",
    "src/schemas.py",
    "src/scoring.py",
    "src/types.py",
    "src/utils.py",
]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def build() -> None:
    bundled = {name: (ROOT / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    bootstrap = f'''# This bundle makes the notebook runnable when opened by itself in Colab.
import json, sys
from pathlib import Path

try:
    import google.colab  # type: ignore[import-not-found]
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    PROJECT_ROOT = Path("/content/reference_free_adaptive_rag_project")
    _BUNDLED_FILES = json.loads({json.dumps(json.dumps(bundled))})
    for relative_path, content in _BUNDLED_FILES.items():
        destination = PROJECT_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
else:
    roots = [Path.cwd(), Path.cwd().parent]
    PROJECT_ROOT = next((path for path in roots if (path / "src" / "experiments.py").exists()), None)
    if PROJECT_ROOT is None:
        raise RuntimeError("Run locally from the project root or notebooks directory")
sys.path.insert(0, str(PROJECT_ROOT))
print("Project root:", PROJECT_ROOT)
'''

    cells = [
        markdown("# Reference-Free Adaptive Retrieval Tuning for RAG\n\n"
                 "A complete, self-contained Google Colab experiment implementing the supplied methodology. "
                 "Gold SQuAD answers remain isolated until post-selection evaluation."),
        markdown("## 1. Setup & dependency installation\n\nColab supplies CUDA-enabled PyTorch. This cell installs the remaining pinned-compatible research stack."),
        code('''%pip install -q "transformers>=4.41,<5" "datasets>=2.19,<5" "sentence-transformers>=3.0,<6" "faiss-cpu>=1.8,<2" "accelerate>=0.31,<2" "numpy>=1.26,<3" "pandas>=2.0,<3" "scikit-learn>=1.4,<2" "scipy>=1.12,<2" "matplotlib>=3.8,<4" "seaborn>=0.13,<1" "PyYAML>=6.0,<7" "tqdm>=4.66,<5" "sentencepiece>=0.2,<1"'''),
        markdown("## 2. Imports & configuration\n\nThe notebook embeds the tested `src/` package, so uploading this notebook alone is sufficient in Colab."),
        code(bootstrap),
        code('''# Change these values only; no pipeline rewrite is needed for scaling.
SEED = 42
NUM_QUESTIONS = 20              # Change to 100, 250, or 500 for larger runs.
RUN_BASELINE = True
RUN_ADAPTIVE = True
RUN_ORACLE = True
RUN_ABLATIONS = True
INCLUDE_BGE = True              # False gives the four-candidate MiniLM debug space.
USE_GOOGLE_DRIVE_CACHE = False  # Optional persistent checkpoints across Colab sessions.
CHECKPOINT_EVERY = 10

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
GENERATOR_MODEL = "microsoft/Phi-3-mini-4k-instruct"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

from src.config import ExperimentConfig
from src.experiments import ExperimentRunner

config = ExperimentConfig(
    seed=SEED,
    num_questions=NUM_QUESTIONS,
    generator_model_id=GENERATOR_MODEL,
    nli_model_id=NLI_MODEL,
    embedding_models={"minilm": EMBEDDING_MODEL, "bge_small": BGE_MODEL},
    uncertainty_embedding_model_id=EMBEDDING_MODEL,
    include_bge=INCLUDE_BGE,
    run_baseline=RUN_BASELINE,
    run_adaptive=RUN_ADAPTIVE,
    run_oracle=RUN_ORACLE,
    run_ablations=RUN_ABLATIONS,
    use_google_drive_cache=USE_GOOGLE_DRIVE_CACHE,
    checkpoint_every=CHECKPOINT_EVERY,
    cache_dir=str(PROJECT_ROOT / "cache"),
    results_dir=str(PROJECT_ROOT / "results"),
)
config.as_dict()'''),
        markdown("## 3. Environment & reproducibility\n\nSeeds are applied to Python, NumPy, PyTorch, and CUDA. Stochastic uncertainty remains reproducible through per-query, per-candidate, per-sample seeds."),
        code('''import time
_experiment_started = time.perf_counter()
runner = ExperimentRunner(config)
runner.prepare()'''),
        markdown("## 4. Dataset loading\n\nA deterministic sample is drawn from the SQuAD 1.1 validation split and cached by dataset, split, N, and seed."),
        markdown("## 5. Hidden-reference separation\n\n`RAGExample` has no answer field. Gold answers are stored in a separate evaluation-only mapping and never passed into retrieval, R/U/C, normalization, J, or selection."),
        code('''assert all("answer" not in example.as_dict() and "answers" not in example.as_dict() for example in runner.rag_examples)
assert not hasattr(runner, "gold")
print(f"Reference-free questions: {len(runner.rag_examples)}")
print("Gold records remain sealed on disk until the evaluation stage.")'''),
        markdown("## 6. Token-aware chunking\n\nEach embedding model's own tokenizer creates 256- or 512-token chunks with 50 content-token overlap; special tokens are included in the size ceiling."),
        markdown("## 7. Embeddings\n\nMiniLM and BGE-small vectors are L2-normalized. The same fixed MiniLM evaluator embeds A1/A2/A3 for semantic uncertainty."),
        markdown("## 8. FAISS retrieval\n\n`IndexFlatIP` over unit vectors gives exact cosine similarity. Indexes are shared across candidates that differ only in Top-K."),
        markdown("## 9. Generator setup\n\nPhi-3 Mini runs in FP16 on CUDA with `device_map='auto'`, chat formatting, a grounded prompt, and inference mode."),
        markdown("## 10. Fixed RAG baseline\n\nMiniLM + 512 tokens + K=3 uses deterministic generation and the same retrieval implementation as every adaptive candidate."),
        code('''baseline_records = runner.run_baseline() if RUN_BASELINE else []
print(f"Baseline complete: {len(baseline_records)} questions")'''),
        markdown("## 11. Candidate configuration construction\n\nThe full experiment evaluates 2 embedding models x 2 chunk sizes x 2 Top-K values = 8 candidates. Set `INCLUDE_BGE=False` for the four-candidate MiniLM phase."),
        code('''[candidate.as_dict() for candidate in config.candidates()]'''),
        markdown("## 12. Retrieval Confidence R\n\nFor retrieved cosine scores $s_i$, the methodology defines $R = \\frac{1}{K}\\sum_i s_i$."),
        markdown("## 13. Semantic Uncertainty U\n\nThree stochastic answers are embedded. With the three pairwise cosines, $S$ is their mean and $U=1-S$."),
        markdown("## 14. Answer-Context Consistency C\n\nFor each sentence in A1, DeBERTa NLI scores every retrieved chunk as premise. The per-sentence maximum entailment probability is averaged."),
        markdown("## 15. Within-query normalization\n\nR, 1-U, and C are min-max normalized only across the current query's candidates. A tied signal receives neutral 0.5 for every candidate."),
        markdown("## 16. Adaptive J scoring\n\nThe exact equal-weight objective is $J = [R' + (1-U)' + C']/3$."),
        markdown("## 17. Configuration selection\n\nThe reference-free selector takes argmax J. Exact ties use fixed methodology candidate order; no evaluation field is accepted."),
        markdown("## 18. Final Adaptive RAG generation\n\nAfter selection, one deterministic final answer is generated from the selected candidate's context."),
        code('''candidate_records, adaptive_records = (runner.run_adaptive() if RUN_ADAPTIVE else ([], []))
print(f"Adaptive complete: {len(adaptive_records)} questions, {len(candidate_records)} candidate rows")'''),
        markdown("## 19. Gold-reference loading for evaluation\n\nOnly now are the already-isolated references joined to generated outputs."),
        markdown("## 20. EM/F1 evaluation\n\nOfficial SQuAD normalization is used, with the best score across multiple valid answers."),
        markdown("## 21. Retrieval evaluation\n\nHit@K checks whether any normalized gold answer span occurs in any retrieved chunk; relevant-document Hit@K is also logged."),
        markdown("## 22. Reference-based oracle\n\nThe post-hoc oracle chooses the candidate answer with maximum hidden-reference F1, ties by EM and fixed candidate order. It cannot influence adaptive outputs."),
        markdown("## 23. Selection regret & correlation\n\nRegret is oracle-best candidate F1 minus adaptive-selected candidate F1. Spearman correlations compare every intrinsic signal with EM/F1."),
        markdown("## 24. Ablations\n\nR; 1-U; C; R+(1-U); R+C; (1-U)+C; and the full objective reuse cached candidate measurements without new generations."),
        markdown("## 25. Embedding-model comparison\n\nMiniLM and BGE-small are grouped with controlled chunk/Top-K/generator settings for retrieval, quality, latency, and selection comparisons."),
        markdown("## 26. Scaling experiments\n\nChange only `NUM_QUESTIONS` to 100, 250, or 500. Atomic checkpoints resume every `CHECKPOINT_EVERY` completed questions."),
        markdown("## 27. Results export\n\nEvaluation, oracle, regret, ablation, embedding comparison, correlations, summaries, environment data, and resolved configuration are exported automatically."),
        code('''outputs = runner.evaluate_and_export(baseline_records, candidate_records, adaptive_records)
outputs["wall_clock_seconds"] = time.perf_counter() - _experiment_started
outputs["summary"]["total_experiment_runtime_seconds"] = outputs["wall_clock_seconds"]
from src.cache import atomic_write_json
atomic_write_json(runner.results_dir / "summary_metrics.json", outputs["summary"])
print("Results directory:", runner.results_dir)'''),
        markdown("## 28. Visualizations\n\nThe exported plots cover system quality, selection distribution, signal distributions, correlation, ablations, and latency."),
        code('''from IPython.display import Image, display
for plot_path in sorted((runner.results_dir / "plots").glob("*.png")):
    print(plot_path.name)
    display(Image(filename=str(plot_path)))'''),
        markdown("## 29. Final summary\n\nThis concise record is also saved as `summary_metrics.json`."),
        code('''import json
print(json.dumps(outputs["summary"], indent=2))'''),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "reference_free_adaptive_rag": {
                "source_sha256": {
                    name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                    for name, content in bundled.items()
                }
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
