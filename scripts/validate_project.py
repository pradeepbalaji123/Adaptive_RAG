"""Static validation for source, notebook, methodology guards, and schemas."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig  # noqa: E402
from src.schemas import REQUIRED_COLUMNS  # noqa: E402
from src.scoring import normalize_candidate_signals, select_adaptive_candidate  # noqa: E402
from src.types import RAGExample  # noqa: E402


REQUIRED_FILES = [
    "README.md",
    "PROJECT_SPEC.md",
    "requirements.txt",
    "configs/experiment.yaml",
    "notebooks/reference_free_adaptive_rag_colab.ipynb",
    "docs/METHODOLOGY_MAPPING.md",
    "docs/IMPLEMENTATION_CONTRACT.md",
    "docs/Proposed_Methodology_Reference_Free_Adaptive_RAG.pdf",
    "src/data.py",
    "src/chunking.py",
    "src/embeddings.py",
    "src/retrieval.py",
    "src/generation.py",
    "src/scoring.py",
    "src/evaluation.py",
    "src/experiments.py",
    "tests/test_core.py",
]


def validate_python_sources() -> int:
    count = 0
    placeholder_markers = ["TO" + "DO", "NotImplemented" + "Error"]
    for path in sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        if any(marker in source for marker in placeholder_markers):
            raise AssertionError(f"Placeholder marker found in {path}")
        count += 1
    return count


def import_all_source_modules() -> int:
    modules = sorted(path.stem for path in (ROOT / "src").glob("*.py") if path.stem != "__init__")
    for module in modules:
        importlib.import_module(f"src.{module}")
    return len(modules)


def validate_notebook() -> int:
    path = ROOT / "notebooks" / "reference_free_adaptive_rag_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and isinstance(notebook["cells"], list)
    headings = [
        line for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        for line in str(cell["source"]).splitlines() if line.startswith("## ")
    ]
    assert len(headings) == 29, f"Expected 29 requested notebook sections, got {len(headings)}"
    code_text = "\n".join(str(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for marker in ["NUM_QUESTIONS = 20", "torch.cuda.is_available()", "runner.run_baseline()", "runner.run_adaptive()", "runner.evaluate_and_export"]:
        assert marker in code_text, f"Notebook is missing {marker}"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = str(cell["source"])
        if source.lstrip().startswith(("%", "!")):
            continue
        ast.parse(source, filename=f"{path.name}:code-cell")
    hashes = notebook["metadata"]["reference_free_adaptive_rag"]["source_sha256"]
    for relative_path, expected in hashes.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected, f"Notebook bundle is stale for {relative_path}"
    return len(notebook["cells"])


def validate_reference_boundary() -> None:
    assert "answer" not in RAGExample.__dataclass_fields__
    assert "answers" not in RAGExample.__dataclass_fields__
    for function in [normalize_candidate_signals, select_adaptive_candidate]:
        parameters = set(inspect.signature(function).parameters)
        assert not parameters.intersection({"gold", "answers", "gold_answers", "references"})
    experiments_source = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
    assert "self.gold" not in experiments_source


def validate_methodology_configuration() -> None:
    config = ExperimentConfig()
    assert len(config.candidates()) == 8
    assert config.uncertainty_samples == 3
    assert config.chunk_sizes == (256, 512)
    assert config.top_k_values == (3, 5)
    assert config.chunk_overlap == 50
    assert config.baseline_candidate().candidate_id == "minilm_c512_k3"
    assert {"baseline_results.csv", "candidate_results.csv", "adaptive_results.csv"}.issubset(REQUIRED_COLUMNS)


def validate_runtime_contract() -> None:
    generation = (ROOT / "src" / "generation.py").read_text(encoding="utf-8")
    embeddings = (ROOT / "src" / "embeddings.py").read_text(encoding="utf-8")
    retrieval = (ROOT / "src" / "retrieval.py").read_text(encoding="utf-8")
    experiments = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
    for marker in ["torch.float16", '"device_map": "auto"', "model.eval()", "torch.inference_mode()", "do_sample"]:
        assert marker in generation
    assert "normalize_embeddings=True" in embeddings
    assert "faiss.IndexFlatIP" in retrieval
    assert "torch.cuda.is_available()" in experiments and "torch.cuda.get_device_name(0)" in experiments


def validate_preserved_sources() -> None:
    specification = (ROOT / "PROJECT_SPEC.md").read_text(encoding="utf-8")
    assert "Reference-Free Adaptive Retrieval Tuning" in specification
    methodology = (ROOT / "docs" / "Proposed_Methodology_Reference_Free_Adaptive_RAG.pdf").read_bytes()
    assert methodology.startswith(b"%PDF-") and len(methodology) > 10_000


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise AssertionError(f"Required project files missing: {missing}")
    source_count = validate_python_sources()
    imported_count = import_all_source_modules()
    cell_count = validate_notebook()
    validate_reference_boundary()
    validate_methodology_configuration()
    validate_runtime_contract()
    validate_preserved_sources()
    print(
        f"Validated {source_count} Python files, imported {imported_count} source modules, "
        f"checked {cell_count} notebook cells, methodology config, schemas, and reference boundary."
    )


if __name__ == "__main__":
    main()
