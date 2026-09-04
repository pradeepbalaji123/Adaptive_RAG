# Reference-Free Adaptive Retrieval Tuning for RAG

This repository is a clean-room implementation of the supplied methodology for selecting a RAG retrieval configuration without using a reference answer. It runs a fixed baseline, the proposed reference-free selector, and a post-hoc reference-based oracle on a deterministic SQuAD 1.1 subset.

Source provenance is preserved in `PROJECT_SPEC.md` and `docs/Proposed_Methodology_Reference_Free_Adaptive_RAG.pdf`. The resolved operational choices are recorded in `docs/IMPLEMENTATION_CONTRACT.md`, with requirement-to-code traceability in `docs/METHODOLOGY_MAPPING.md`.

The primary deliverable is the self-contained Colab notebook at `notebooks/reference_free_adaptive_rag_colab.ipynb`. The notebook embeds the tested `src/` package, so it can be uploaded to Colab by itself and run without cloning this repository.

## Methodology implemented

For each query and candidate configuration:

1. Retrieve Top-K chunks with cosine similarity (L2-normalized embeddings plus FAISS inner product).
2. Compute retrieval confidence as the mean similarity: `R = mean(s_1, ..., s_K)`.
3. Generate exactly three stochastic grounded answers, A1/A2/A3.
4. Embed those answers with fixed MiniLM, average their three pairwise cosine similarities to obtain S, and compute `U = 1 - S`.
5. Split A1 into sentences. For each sentence, take the maximum DeBERTa NLI entailment probability over retrieved chunks, then average those maxima to obtain C.
6. Min-max normalize R, semantic confidence `(1-U)`, and C within that query's candidate set. If a signal is identical for all candidates, every candidate receives the neutral normalized value 0.5 for that signal.
7. Compute `J = [R' + (1-U)' + C'] / 3` and select its argmax. Exact ties use fixed candidate order.
8. Generate one deterministic final answer from the selected context.

The source documents say C uses “the generated answer” but do not state which of the three uncertainty samples. This implementation uses A1 for C and for comparable post-hoc candidate/oracle scoring. All three samples are used for U. The selected configuration still receives the separately required deterministic final generation.

## Reference-free boundary

`src.types.RAGExample` cannot hold gold answers. The dataset loader immediately creates two separate partitions:

- `rag_examples`: question/context records accepted by the adaptive pipeline;
- `gold`: evaluation-only `GoldRecord` objects.

`ExperimentRunner.prepare` retains only `rag_examples`; it has no gold attribute. The selector rejects candidate records containing gold/reference, EM, or F1 fields. `load_gold_for_evaluation` opens the sealed partition only inside `evaluate_and_export`, after adaptive answers and selections already exist.

## Candidate systems

- Fixed RAG: MiniLM, 512-token chunks, overlap 50, K=3, deterministic Phi-3.
- Adaptive debug space: MiniLM x chunk sizes {256, 512} x K {3, 5}.
- Full adaptive space: debug space plus the same four BGE-small candidates.
- Oracle: post-hoc maximum candidate F1 using hidden SQuAD answers; it never influences adaptive selection.

Models:

- Generator: `microsoft/Phi-3-mini-4k-instruct`
- Primary retriever and uncertainty evaluator: `sentence-transformers/all-MiniLM-L6-v2`
- Comparative retriever: `BAAI/bge-small-en-v1.5`
- NLI evaluator: `cross-encoder/nli-deberta-v3-base`

## Run in Google Colab

1. Upload and open `notebooks/reference_free_adaptive_rag_colab.ipynb` in Google Colab.
2. Select **Runtime -> Change runtime type -> GPU**.
3. Select **Runtime -> Run all**.

The default experiment evaluates 20 questions with all eight candidates. It performs 26 Phi-3 generations per question: one fixed-baseline generation, 24 uncertainty samples, and one adaptive final generation. The first run downloads several models and is intentionally expensive. Set `INCLUDE_BGE=False` for the four-candidate MiniLM phase, or set `USE_GOOGLE_DRIVE_CACHE=True` to preserve caches and checkpoints across Colab sessions.

To scale, change only `NUM_QUESTIONS` near the top of the notebook to 100, 250, or 500. Completed questions resume from atomic checkpoints every `CHECKPOINT_EVERY` questions.

## Run as a project

```bash
python -m pip install -r requirements.txt
python scripts/run_experiment.py --config configs/experiment.yaml
```

Useful overrides:

```bash
python scripts/run_experiment.py --num-questions 20 --minilm-only
python scripts/run_experiment.py --num-questions 100 --cache-dir cache --results-dir results
```

An NVIDIA GPU is strongly recommended. CPU execution is supported for correctness but Phi-3 generation will be very slow.

## Architecture

```text
SQuAD 1.1
  -> deterministic subset
  -> immediate RAG/gold separation
  -> deduplicated corpus
  -> tokenizer-specific 256/512 chunks
  -> normalized MiniLM/BGE embeddings
  -> cached FAISS IndexFlatIP indexes
  -> fixed baseline
  -> per-query candidate retrieval + A1/A2/A3 + R/U/C
  -> within-query normalization + J argmax
  -> deterministic final adaptive answer
  -> evaluation-only gold join
  -> oracle, regret, correlations, ablations, embedding comparison
  -> CSV/JSON/PNG exports
```

`src/experiments.py` owns this flow. Models are loaded lazily and reused. Chunk data, normalized embeddings, FAISS indexes, retrievals, sampled generations, deterministic generations, NLI scores, semantic scores, and question-level progress are cached.

## Results

The run creates:

```text
results/
  baseline_results.csv
  candidate_results.csv
  per_query_results.csv
  adaptive_results.csv
  evaluation_results.csv
  oracle_results.csv
  selection_regret.csv
  ablation_results.csv
  ablation.csv
  embedding_comparison.csv
  correlation_results.csv
  selection_distribution.csv
  summary.csv
  summary_metrics.json
  config_resolved.json
  environment.json
  plots/
    system_quality.png
    selected_configuration_distribution.png
    signal_distributions.png
    correlation_heatmap.png
    ablation_comparison.png
    latency_comparison.png
```

Candidate CSV rows retain retrieved chunk IDs/texts, similarity scores, all three answers, raw and normalized signals, J, timings, token counts, and the adaptive selection marker. Gold answers and answer-quality metrics appear only in post-hoc exported copies.

## Verification

Run the inexpensive tests without downloading any model:

```bash
python -m unittest discover -s tests -v
python scripts/build_notebook.py
python scripts/validate_project.py
```

These cover deterministic selection, reference isolation, chunk limits/overlap, exact Top-K handling, normalized-vector retrieval assumptions, R/U/C formulas, tied normalization, J, deterministic selection, grounded prompt construction, non-empty generation validation, SQuAD EM/F1, answer-span Hit@K, and export schemas.

## Research notes

- Selection regret compares oracle candidate A1 F1 with the A1 F1 of the adaptively selected candidate. Final adaptive-answer F1 is reported separately in the system comparison.
- BGE and MiniLM use the same unprefixed encoding procedure because the supplied methodology requires controlled, consistent retrieval and does not specify a BGE query instruction.
- NLI uses the official model-card label order fallback `[contradiction, entailment, neutral]` only when the model config lacks descriptive labels.
- A full 20-question Phi-3/NLI run requires Colab GPU execution and model downloads; local validation intentionally exercises only deterministic, inexpensive components.
