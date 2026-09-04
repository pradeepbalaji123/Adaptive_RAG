# Methodology-to-Code Mapping

This checklist maps every supplied methodology component to executable code and tests.

| Methodology component | Implementation | Verification |
|---|---|---|
| Dataset preparation | `src.data.load_squad_subset`, `create_corpus` | deterministic subset and record-count checks |
| Hidden-reference separation | `src.types.RAGExample`, `GoldRecord`, `assert_reference_free_examples`; evaluation-only join in `src.evaluation.attach_gold_evaluation` | explicit leak-rejection test |
| Token chunking | `src.chunking.chunk_documents` | token ceiling and exact overlap test |
| Embedding models | `src.embeddings.SentenceEmbeddingRegistry` | unit-norm guard; MiniLM/BGE configuration test |
| FAISS indexes and cosine retrieval | `src.retrieval.FaissIndexManager` | exact Top-K and similarity-score test |
| Fixed baseline | `src.experiments.ExperimentRunner._baseline_one`, `run_baseline` | required export schema |
| Candidate space | `src.config.ExperimentConfig.candidates` | four- and eight-candidate tests |
| Grounded generation | `src.generation.build_grounded_messages`, `Phi3Generator` | injection boundary and non-empty answer tests |
| Retrieval Confidence R | `src.scoring.compute_retrieval_confidence` | exact mean formula test |
| Semantic Uncertainty U | `estimate_semantic_uncertainty`, `semantic_uncertainty_from_embeddings` | exact three-pair formula test |
| Answer-Context Consistency C | `NLIScorer`, `compute_answer_context_consistency` | sentence-max/chunk then mean test |
| Normalization | `min_max_normalize`, `normalize_candidate_signals` | range and neutral tie tests |
| Objective J | `compute_reference_free_score` | exact equal-weight formula test |
| Adaptive selection | `select_adaptive_candidate` | deterministic tie and evaluation-field rejection tests |
| Final adaptive answer | `ExperimentRunner._adaptive_one` | selected context plus deterministic generation path |
| SQuAD evaluation | `normalize_answer`, `evaluate_answer`, `attach_gold_evaluation` | synthetic EM/F1 test |
| Retrieval evaluation | `src.evaluation.retrieval_hit`; relevant-document join | synthetic answer-span Hit@K test |
| Oracle | `select_oracle_candidates` | invoked only after gold evaluation |
| Selection regret | `selection_regret_records` | exported per query and summarized |
| Correlation | `correlation_analysis` | Spearman R/U/(1-U)/C/J against EM/F1 |
| Ablations | `run_ablations`, `ABLATION_SIGNALS` | seven specified combinations reuse cached rows |
| Embedding comparison | `embedding_comparison` | grouped MiniLM/BGE quality, retrieval, latency, selection |
| Caching/checkpoints | `src.cache.DiskCache`, `CheckpointStore`; index caches in `src.retrieval` | atomic writes and configuration-hash guarding |
| Research plots | `src.plotting.create_research_plots` | six labeled PNG exports |
| Colab Run All | `notebooks/reference_free_adaptive_rag_colab.ipynb` | JSON and embedded-source validation |

## Evaluation boundary audit

The call graph is intentionally one-way:

```text
load_squad_subset
  -> rag_examples ---------------------> prepare/retrieve/generate/R/U/C/J/select
  -> gold evaluation-only mapping -----> attach_gold_evaluation -> oracle/metrics
```

`ExperimentRunner.run_adaptive` has access only to `RAGExample` records. Although the runner owns the separate gold mapping for the later export stage, neither `_candidate_record`, `_adaptive_one`, normalization, nor selection accepts it as an argument. The selector also rejects evaluation fields at runtime.

## Operational ambiguity record

The methodology requires A1/A2/A3 for U and describes C using a singular generated answer, without specifying an aggregation over sampled answers. This implementation uses A1 for C and post-hoc candidate/oracle F1. That keeps C exactly a single answer-context score and avoids adding an unapproved fourth candidate generation. The adaptive winner still receives the required deterministic final generation.

