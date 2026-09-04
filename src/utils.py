"""Reproducibility, environment reporting, and safe GPU cleanup."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import platform
import random
from datetime import datetime, timezone
from typing import Any


def resolve_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def controlled_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, (base_seed,) + parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def reset_peak_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_gpu_memory_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except ImportError:
        pass
    return 0.0


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_report(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "sentence_transformers": _version("sentence-transformers"),
        "faiss": _version("faiss-cpu"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seed": config["seed"],
        "generator_model_id": config["generator_model_id"],
        "embedding_models": config["embedding_models"],
        "nli_model_id": config["nli_model_id"],
    }
