"""Deterministic SQuAD 1.1 loading with immediate gold-reference isolation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .cache import atomic_write_json, canonical_hash
from .types import Document, GoldRecord, RAGExample, assert_reference_free_examples


def _document_id(context: str) -> str:
    return "doc_" + hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]


def deterministic_subset_indices(dataset_size: int, num_questions: int, seed: int) -> list[int]:
    if dataset_size < 0 or num_questions <= 0 or num_questions > dataset_size:
        raise ValueError("Invalid dataset or subset size")
    return random.Random(seed).sample(range(dataset_size), num_questions)


def _subset_paths(
    num_questions: int,
    seed: int,
    dataset_name: str,
    split: str,
    cache_dir: str | Path,
) -> tuple[Path, Path]:
    key = canonical_hash({"dataset": dataset_name, "split": split, "n": num_questions, "seed": seed})
    subset_dir = Path(cache_dir) / "dataset" / key
    return subset_dir / "rag_examples.json", subset_dir / "gold_answers_evaluation_only.json"


def load_squad_subset(
    num_questions: int,
    seed: int,
    dataset_name: str = "rajpurkar/squad",
    split: str = "validation",
    cache_dir: str | Path = "cache",
) -> tuple[list[RAGExample], dict[str, GoldRecord]]:
    """Load a stable subset and return gold records through a separate object.

    The adaptive pipeline receives only the first return value. The raw dataset row
    and its ``answers`` field are not retained in any RAGExample.
    """

    if num_questions <= 0:
        raise ValueError("num_questions must be positive")
    rag_path, gold_path = _subset_paths(num_questions, seed, dataset_name, split, cache_dir)
    if rag_path.exists() and gold_path.exists():
        try:
            rag_payload = json.loads(rag_path.read_text(encoding="utf-8"))
            gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
            examples = [RAGExample(**item) for item in rag_payload]
            gold = {
                item["query_id"]: GoldRecord(
                    query_id=item["query_id"],
                    answers=tuple(item["answers"]),
                    document_id=item["document_id"],
                )
                for item in gold_payload
            }
            assert_reference_free_examples(examples)
            if len(examples) == num_questions and len(gold) == num_questions:
                return examples, gold
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package before loading SQuAD") from exc

    dataset = load_dataset(dataset_name, split=split, cache_dir=str(Path(cache_dir) / "huggingface"))
    if num_questions > len(dataset):
        raise ValueError(f"Requested {num_questions} questions but split contains {len(dataset)}")
    indices = deterministic_subset_indices(len(dataset), num_questions, seed)
    examples: list[RAGExample] = []
    gold: dict[str, GoldRecord] = {}
    for index in indices:
        row = dataset[int(index)]
        try:
            query_id = str(row["id"])
            question = str(row["question"]).strip()
            context = str(row["context"]).strip()
            title = str(row.get("title", ""))
            answer_texts = tuple(str(value).strip() for value in row["answers"]["text"] if str(value).strip())
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed SQuAD row at index {index}") from exc
        if not query_id or not question or not context or not answer_texts:
            raise ValueError(f"Malformed SQuAD row at index {index}")
        document_id = _document_id(context)
        examples.append(RAGExample(query_id, question, document_id, context, title))
        gold[query_id] = GoldRecord(query_id, answer_texts, document_id)

    assert_reference_free_examples(examples)
    atomic_write_json(rag_path, [example.as_dict() for example in examples])
    atomic_write_json(gold_path, [gold[example.query_id].as_dict() for example in examples])
    return examples, gold


def prepare_reference_free_squad_data(
    num_questions: int,
    seed: int,
    dataset_name: str = "rajpurkar/squad",
    split: str = "validation",
    cache_dir: str | Path = "cache",
) -> list[RAGExample]:
    """Return only the RAG side of the split; the runner never retains gold data."""

    examples, evaluation_only_gold = load_squad_subset(
        num_questions, seed, dataset_name, split, cache_dir
    )
    del evaluation_only_gold
    return examples


def load_gold_for_evaluation(
    num_questions: int,
    seed: int,
    dataset_name: str = "rajpurkar/squad",
    split: str = "validation",
    cache_dir: str | Path = "cache",
) -> dict[str, GoldRecord]:
    """Reveal the sealed gold partition only after adaptive inference is complete."""

    _, gold_path = _subset_paths(num_questions, seed, dataset_name, split, cache_dir)
    if not gold_path.exists():
        raise RuntimeError("Gold partition is unavailable; prepare the reference-free dataset first")
    try:
        payload = json.loads(gold_path.read_text(encoding="utf-8"))
        records = {
            item["query_id"]: GoldRecord(
                query_id=item["query_id"],
                answers=tuple(item["answers"]),
                document_id=item["document_id"],
            )
            for item in payload
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("The evaluation-only gold cache is malformed") from exc
    if len(records) != num_questions:
        raise RuntimeError("Evaluation-only gold record count does not match the configured subset")
    return records


def create_corpus(examples: Sequence[RAGExample]) -> list[Document]:
    """Deduplicate selected SQuAD passages while retaining source question IDs."""

    assert_reference_free_examples(examples)
    sources: dict[str, list[str]] = defaultdict(list)
    first: dict[str, RAGExample] = {}
    for example in examples:
        sources[example.document_id].append(example.query_id)
        first.setdefault(example.document_id, example)
    return [
        Document(
            document_id=document_id,
            text=first[document_id].context,
            title=first[document_id].title,
            source_example_ids=tuple(sources[document_id]),
        )
        for document_id in sorted(first)
    ]


def save_corpus(corpus: Sequence[Document], path: str | Path) -> None:
    atomic_write_json(path, [document.as_dict() for document in corpus])
