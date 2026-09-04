# Implemented Research Contract

This file records the concrete contract distilled from the supplied Proposed Methodology PDF and attached project specification. The PDF governs the research algorithm; the project specification governs implementation constraints.

## Primary research rule

Adaptive configuration selection is reference-free. SQuAD answers are inaccessible to retrieval, candidate generation, R, U, C, normalization, J, argmax selection, and final adaptive generation. Gold answers are revealed only during evaluation and oracle construction.

## Required models and data

- Dataset: SQuAD 1.1 (`rajpurkar/squad`), deterministic validation subset, seed 42, default N=20.
- Generator: `microsoft/Phi-3-mini-4k-instruct`.
- Retrievers: `sentence-transformers/all-MiniLM-L6-v2` and comparative `BAAI/bge-small-en-v1.5`.
- Fixed semantic-uncertainty evaluator: MiniLM.
- Consistency evaluator: `cross-encoder/nli-deberta-v3-base` entailment probability.

## Retrieval configurations

- Chunk sizes: 256 and 512 model tokens.
- Fixed overlap: 50 content tokens.
- Top-K values: 3 and 5.
- Debug space: four MiniLM configurations.
- Full space: those four plus four corresponding BGE-small configurations.
- Fixed baseline: MiniLM, chunk 512, K=3.
- Retrieval: L2-normalized vectors with FAISS `IndexFlatIP`, so inner product is cosine similarity.

## Exact signals

- `R = mean(Top-K cosine similarities)`.
- Generate exactly three stochastic answers A1/A2/A3 per candidate.
- Sampling uses `do_sample=True`, temperature 0.7, Top-P 0.9, and 96 new tokens because the methodology fixes stochasticity and equality across candidates but does not prescribe numerical sampling parameters.
- `S = mean(cos(A1,A2), cos(A1,A3), cos(A2,A3))`.
- `U = 1 - S`; semantic confidence is `1-U`.
- C: split A1 into sentences; for each answer sentence compute entailment against every retrieved chunk, take its maximum, then average across answer sentences.
- Normalize R, 1-U, and C with min-max scaling within one query's candidates. A zero-width range maps every value to neutral 0.5.
- `J = [R' + (1-U)' + C'] / 3`.
- Select maximum J; exact ties use fixed candidate order.
- Generate one deterministic final answer from the selected context.

The A1 choice resolves the source documents' only operational ambiguity: they specify three answers for U but refer to a singular generated answer for C without naming which sample. The choice does not introduce reference information.

## Required comparisons and analysis

- Fixed RAG, reference-free adaptive RAG, and post-hoc reference-based oracle.
- Official normalized SQuAD Exact Match and token F1, maximizing over valid references.
- Answer-span Retrieval Hit@K and relevant-document Hit@K.
- NLI faithfulness, latency, LLM generations, token counts, and peak GPU memory.
- Selection regret, zero-regret proportion, and oracle agreement.
- Spearman correlations between intrinsic signals and EM/F1.
- Seven cached ablations: R; 1-U; C; R+(1-U); R+C; (1-U)+C; and all three.
- MiniLM/BGE comparison and adaptive configuration distribution.
- CSV, JSON, and publication-readable PNG exports.

## Execution contract

- Google Colab, Python 3, NVIDIA CUDA preferred.
- FP16 and automatic device mapping for Phi-3 on CUDA; inference mode and model evaluation mode.
- Controlled stochastic seeds for uncertainty; deterministic final generation.
- Atomic caching and question checkpoints; optional Google Drive persistence.
- A standalone Colab notebook that supports Run All.
- No LangChain, agents, GraphRAG, fine-tuning, reinforcement learning, or hard-coded results.
