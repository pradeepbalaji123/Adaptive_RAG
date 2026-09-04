"""Embedding-tokenizer-aware chunking with exact overlap in content tokens."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from .types import Chunk, Document


def _encode(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False)
    return list(token_ids)


def chunk_documents(
    documents: Sequence[Document],
    tokenizer: Any,
    embedding_key: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Create chunks whose model-tokenized length cannot exceed ``chunk_size``.

    ``chunk_size`` includes model special tokens; overlap counts content tokens.
    This prevents SentenceTransformer from silently truncating nominal 512-token chunks.
    """

    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be positive and overlap non-negative")
    special_count = 0
    if hasattr(tokenizer, "num_special_tokens_to_add"):
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_window = chunk_size - special_count
    if content_window <= overlap:
        raise ValueError("chunk_size minus special tokens must exceed overlap")
    step = content_window - overlap
    chunks: list[Chunk] = []
    for document in documents:
        token_ids = _encode(tokenizer, document.text, add_special_tokens=False)
        if not token_ids:
            continue
        for chunk_index, start in enumerate(range(0, len(token_ids), step)):
            window = token_ids[start : start + content_window]
            if not window:
                break
            text = tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            if not text:
                continue
            encoded_count = len(_encode(tokenizer, text, add_special_tokens=True))
            while encoded_count > chunk_size and window:
                window = window[:-1]
                text = tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                encoded_count = len(_encode(tokenizer, text, add_special_tokens=True))
            if not text:
                continue
            digest = hashlib.sha256(
                f"{embedding_key}|{chunk_size}|{document.document_id}|{chunk_index}|{start}".encode("utf-8")
            ).hexdigest()[:18]
            chunks.append(Chunk(
                chunk_id=f"chunk_{digest}",
                document_id=document.document_id,
                text=text,
                token_count=encoded_count,
                chunk_index=chunk_index,
                embedding_key=embedding_key,
                chunk_size=chunk_size,
                source_example_ids=document.source_example_ids,
            ))
            if start + content_window >= len(token_ids):
                break
    if not chunks:
        raise ValueError("Chunking produced no usable text")
    return chunks

