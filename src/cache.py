"""Small, transparent disk caches and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def canonical_hash(value: Any, length: int = 20) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DiskCache:
    """JSON cache keyed by a canonical hash of complete computation inputs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, namespace: str, key: Any) -> Path:
        return self.root / namespace / f"{canonical_hash(key)}.json"

    def get(self, namespace: str, key: Any) -> Any | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, key: Any, value: Any) -> None:
        atomic_write_json(self._path(namespace, key), value)

    def get_or_compute(self, namespace: str, key: Any, compute: Callable[[], T]) -> T:
        cached = self.get(namespace, key)
        if cached is not None:
            return cached
        value = compute()
        self.set(namespace, key, value)
        return value


class CheckpointStore:
    """Atomic stage checkpoints guarded by the resolved configuration hash."""

    def __init__(self, root: str | Path, config: dict[str, Any]) -> None:
        self.config_hash = canonical_hash(config)
        self.root = Path(root) / "checkpoints" / self.config_hash

    def load(self, stage: str) -> list[dict[str, Any]]:
        path = self.root / f"{stage}.json"
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("config_hash") != self.config_hash:
                return []
            records = payload.get("records", [])
            return records if isinstance(records, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, stage: str, records: list[dict[str, Any]]) -> None:
        atomic_write_json(
            self.root / f"{stage}.json",
            {"config_hash": self.config_hash, "records": records},
        )

