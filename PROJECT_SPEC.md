# Reference-Free Adaptive Retrieval Tuning for RAG Systems

## 1. Project Title

**Reference-Free Adaptive Retrieval Tuning for RAG Systems Using Uncertainty and Consistency Signals**

## 2. Project Goal

Build a compact Retrieval-Augmented Generation (RAG) system that automatically selects the best retrieval configuration for each query without using human-written reference answers during configuration selection.

The system must evaluate candidate retrieval configurations using three intrinsic signals:

- Retrieval Confidence (R)
- Semantic Uncertainty (U)
- Answer-Context Consistency (C)

The configuration with the highest combined reference-free score is selected and used to generate the final answer.

## 3. Core Research Rule

**Ground-truth or reference answers must NEVER be used during retrieval configuration selection.**

Reference answers may be used only after configuration selection for experimental evaluation.

This separation is mandatory because the main research objective is to demonstrate reference-free retrieval tuning.

## 4. User Inputs

The system should accept:

1. A knowledge base or document corpus
2. A natural-language question

For benchmark experiments, use SQuAD 1.1.

For the final demonstration, the knowledge base may be a user-uploaded PDF or text document.

## 5. User Outputs

The system should return:

- Final generated answer
- Selected embedding model
- Selected chunk size
- Selected Top-K value
- Retrieval Confidence score
- Semantic Uncertainty score
- Answer-Context Consistency score
- Final combined reference-free score

## 6. Technology Stack

Use:

- Python
- Google Colab compatible
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- Scikit-learn
- Optional Gradio interface after the research pipeline is complete

Do not use unnecessary frameworks such as LangChain, multi-agent systems, GraphRAG, reinforcement learning, knowledge graphs, or fine-tuning.

## 7. Dataset

Initial dataset:

**SQuAD 1.1**

Use a small subset during development:

- 20 questions for debugging
- 100 questions for preliminary experiments
- 200-500 questions for final evaluation if computation permits

Each record should logically contain:

- query_id
- question
- context/document
- gold_answer
- source document identifier

The gold answer must be isolated from the adaptive-selection pipeline and exposed only to the evaluation module.

## 8. Models

### Generator

Use:

**microsoft/Phi-3-mini-4k-instruct**

The generator should answer using only the retrieved context.

### Embedding Models

Initial development:

**sentence-transformers/all-MiniLM-L6-v2**

After the complete pipeline works, add:

**BAAI/bge-small-en-v1.5**

### NLI Model

Use a pretrained NLI model such as:

**cross-encoder/nli-deberta-v3-base**

Use the entailment probability as the answer-context consistency signal.

## 9. Candidate Retrieval Configurations

Initial debugging configuration space:

- Embedding model: MiniLM only
- Chunk sizes: 256, 512
- Top-K: 3, 5
- Chunk overlap: fixed at 50

This produces 4 candidate configurations.

After the system works, add BGE-small:

- MiniLM + chunk 256 + K=3
- MiniLM + chunk 256 + K=5
- MiniLM + chunk 512 + K=3
- MiniLM + chunk 512 + K=5
- BGE-small + chunk 256 + K=3
- BGE-small + chunk 256 + K=5
- BGE-small + chunk 512 + K=3
- BGE-small + chunk 512 + K=5

Final candidate space: 8 configurations.

## 10. Offline Preparation

Before answering queries:

1. Load the document corpus.
2. Create 256-token chunks with fixed overlap.
3. Create 512-token chunks with fixed overlap.
4. Generate embeddings for each chunk.
5. Build FAISS indexes for each chunk-size and embedding-model combination.
6. Store chunk metadata including document ID and chunk ID.

The following indexes should eventually exist:

- MiniLM + 256
- MiniLM + 512
- BGE-small + 256
- BGE-small + 512

Top-K does not require a separate index.

## 11. Retrieval Module

Implement:

```python
retrieve(question, embedding_model, chunk_size, top_k)
```

Return:

- Retrieved chunks
- Cosine similarity scores
- Chunk metadata

The same retrieval procedure must be used consistently for all configurations.

## 12. Retrieval Confidence (R)

The initial retrieval confidence metric should be simple and reproducible.

For retrieved similarity scores:

\[
s_1, s_2, ..., s_k
\]

compute:

\[
R = \frac{1}{K}\sum_{i=1}^{K}s_i
\]

Therefore:

```python
R = mean(retrieved_similarity_scores)
```

Higher R indicates stronger semantic relevance between the query and retrieved evidence.

Do not add complicated ranking-separation terms until the initial prototype is complete.

## 13. Response Generation

Use a fixed grounded prompt for every candidate configuration.

Example structure:

```text
You are given retrieved evidence.

Answer the question using only the supplied evidence.
If the evidence is insufficient, state that the answer
cannot be determined from the context.

CONTEXT:
{retrieved_context}

QUESTION:
{question}

ANSWER:
```

For each candidate configuration:

Generate 3 sampled answers:

- A1
- A2
- A3

Use the same generation settings for all configurations.

After the best configuration is selected, generate one final deterministic answer from the selected context.

## 14. Semantic Uncertainty (U)

Generate three candidate answers for the same query and retrieved context.

Embed A1, A2 and A3 using one fixed evaluator embedding model.

Compute pairwise semantic similarities:

\[
sim(A_1,A_2), sim(A_1,A_3), sim(A_2,A_3)
\]

Semantic consistency:

\[
S = \frac{
sim(A_1,A_2)+sim(A_1,A_3)+sim(A_2,A_3)
}{3}
\]

Semantic uncertainty:

\[
U = 1-S
\]

Therefore:

```python
semantic_confidence = 1 - U
```

High agreement among answers means low uncertainty.

## 15. Answer-Context Consistency (C)

Use the NLI model to determine whether the generated answer is supported by the retrieved evidence.

Conceptually:

```text
Premise    = retrieved context
Hypothesis = generated answer
```

Use the NLI entailment probability as:

\[
C = P(entailment)
\]

Higher C means stronger evidence support.

For a better implementation:

1. Split the generated answer into sentences.
2. Compare each answer sentence against each retrieved chunk.
3. For each answer sentence, take the maximum entailment probability among the chunks.
4. Average these values across answer sentences.

This produces the final context-consistency score.

## 16. Signal Normalization

For every query, calculate R, semantic confidence `(1-U)`, and C for all candidate configurations.

Normalize each signal across the candidate configurations using min-max normalization:

\[
x' = \frac{x-\min(x)}{\max(x)-\min(x)}
\]

Handle equal-value cases safely to avoid division by zero.

## 17. Reference-Free Objective

After normalization, calculate:

\[
J = \frac{R' + (1-U)' + C'}{3}
\]

Use equal weights in the initial implementation.

Do not tune the weights using benchmark reference answers.

The selected configuration is:

\[
C^* = \arg\max_C J(C)
\]

This selection must happen without using the gold answer.

## 18. Adaptive RAG Flow

For every user question:

1. Iterate through all candidate configurations.
2. Retrieve the Top-K chunks.
3. Compute Retrieval Confidence R.
4. Generate three sampled answers.
5. Compute Semantic Uncertainty U.
6. Compute Answer-Context Consistency C.
7. Normalize R, `(1-U)`, and C across configurations.
8. Compute combined score J.
9. Select the configuration with maximum J.
10. Generate one deterministic final answer using the selected context.
11. Return the answer, selected configuration, and reliability scores.

## 19. Core Functions

The implementation should contain functions equivalent to:

```python
load_dataset()
create_corpus()
chunk_documents()
build_indexes()
retrieve()
generate_answer()
retrieval_confidence()
semantic_uncertainty()
context_consistency()
normalize_signals()
calculate_combined_score()
select_best_configuration()
evaluate_answer()
run_experiment()
adaptive_rag()
```

Keep functions simple, modular and testable.

## 20. Baselines

Compare exactly three systems.

### Baseline 1: Fixed RAG

Use one fixed configuration for every query.

Recommended initial fixed configuration:

- MiniLM
- Chunk size = 512
- Top-K = 3

### Baseline 2: Proposed Reference-Free Adaptive RAG

Select configuration using only:

- R
- `(1-U)`
- C

No reference answer.

### Baseline 3: Reference-Based Oracle

Try all candidate configurations and evaluate their generated answers using the hidden gold answer.

Select the configuration with the highest actual answer-quality score.

This is an upper/reference baseline and must not influence the proposed system.

## 21. Performance Metrics

Evaluate using:

### Exact Match (EM)

Measures exact normalized answer correctness.

### Token-Level F1

Measures token overlap between generated answer and benchmark reference answer.

### Retrieval Hit@K

Measures whether relevant source evidence appears among the retrieved chunks.

### Answer Faithfulness / NLI

Measures whether generated claims are supported by retrieved evidence.

### Selection Regret

\[
Selection\ Regret =
F1_{oracle-best} - F1_{proposed-selected}
\]

Lower regret means the reference-free selector chooses configurations closer to the true optimum.

### Latency

Measure:

- Average query latency
- Retrieval time
- Generation time
- Number of LLM generations
- Tokens processed

## 22. Correlation Analysis

For every candidate configuration record:

- Reference-free J score
- Actual hidden-ground-truth F1

Calculate Spearman correlation between J and actual F1.

A positive correlation supports the hypothesis that the proposed reference-free surrogate objective tracks true answer quality.

## 23. Ablation Study

Evaluate:

- R only
- `(1-U)` only
- C only
- R + `(1-U)`
- R + C
- `(1-U)` + C
- R + `(1-U)` + C

Compare answer quality and selection regret.

The full three-signal method should ideally provide the strongest or most stable performance.

## 24. Result Logging

Save a CSV containing at least:

```text
query_id
configuration_id
embedding_model
chunk_size
top_k
retrieval_confidence
semantic_uncertainty
semantic_confidence
context_consistency
combined_score
generated_answer
selected
latency
gold_answer
exact_match
f1
retrieval_hit
```

Also save:

- summary.csv
- ablation.csv

All experiments should be reproducible.

## 25. Recommended Project Structure

```text
reference-free-adaptive-rag/
│
├── data/
│   ├── corpus.json
│   └── questions.json
│
├── indexes/
│   ├── minilm_256.index
│   ├── minilm_512.index
│   ├── bge_256.index
│   └── bge_512.index
│
├── configs/
│   └── experiment.yaml
│
├── src/
│   ├── dataset.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexer.py
│   ├── retrieval.py
│   ├── generator.py
│   ├── signals.py
│   ├── optimizer.py
│   ├── evaluation.py
│   └── experiment.py
│
├── notebooks/
│   └── adaptive_rag.ipynb
│
├── results/
│   ├── per_query_results.csv
│   ├── summary.csv
│   └── ablation.csv
│
├── docs/
│
├── app.py
├── requirements.txt
├── PROJECT_SPEC.md
└── README.md
```

During early development, keep most implementation inside the Colab notebook. Split code into modules only after the pipeline works.

## 26. Implementation Order

Implement strictly in this order:

1. Load 20 SQuAD questions.
2. Build the document corpus.
3. Implement chunking.
4. Build MiniLM FAISS indexes for chunk sizes 256 and 512.
5. Implement retrieval.
6. Load Phi-3 Mini.
7. Make baseline RAG successfully answer questions.
8. Add four configurations using MiniLM.
9. Implement Retrieval Confidence R.
10. Implement three-generation Semantic Uncertainty U.
11. Implement NLI consistency C.
12. Implement signal normalization.
13. Implement combined J score.
14. Implement best-configuration selection.
15. Evaluate selected answers using hidden SQuAD references.
16. Run successfully on 20 questions.
17. Add BGE-small and expand to eight configurations.
18. Run 100-500 questions.
19. Compare Fixed RAG, Proposed Adaptive RAG and Oracle.
20. Run ablation study.
21. Export final results.
22. Add optional Gradio UI only after experiments are complete.

## 27. Development Constraints

The implementation must:

- Remain Google Colab compatible.
- Avoid unnecessary abstraction.
- Be understandable for a college project presentation.
- Avoid training or fine-tuning models.
- Cache chunks, embeddings, FAISS indexes and generated answers where practical.
- Use reproducible random seeds.
- Keep generation parameters identical across candidate configurations.
- Separate adaptive selection from benchmark evaluation.
- Never silently use the gold answer inside the adaptive pipeline.

## 28. Optional Final Demo

The final Gradio interface may accept:

- Uploaded PDF/document
- User question

It should display:

- Final answer
- Selected embedding model
- Selected chunk size
- Selected Top-K
- Retrieval Confidence
- Semantic Uncertainty
- Answer-Context Consistency
- Combined reference-free score

The demo is secondary. Research experiments and evaluation must be completed first.

## 29. Main Research Question

**Can retrieval confidence, semantic uncertainty, and answer-context consistency act as a reference-free surrogate objective for selecting RAG retrieval configurations without human-written ideal answers?**

## 30. Expected Outcome

The project should demonstrate whether the proposed reference-free selector can achieve answer quality close to a reference-based oracle while eliminating the need for domain-specific ideal answers during retrieval tuning.

The central contribution is not a new LLM or retriever. It is the adaptive selection mechanism that uses complementary retrieval-, generation-, and evidence-grounding signals to tune the RAG pipeline without reference answers.
