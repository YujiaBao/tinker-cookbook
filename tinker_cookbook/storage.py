"""Storage abstraction layer for tinker-cookbook.

Provides a ``Storage`` protocol that abstracts file I/O operations,
with a ``LocalStorage`` implementation for local filesystem access.
Cloud backends (S3, GCS) can be added without changing any reader/writer code.

Usage::

    storage = LocalStorage("/path/to/data")
    storage.write("runs/001/config.json", json.dumps(config).encode())
    data = storage.read("runs/001/config.json")
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageStat:
    """File metadata returned by ``Storage.stat()``."""

    size: int
    mtime: float  # seconds since epoch


@runtime_checkable
class Storage(Protocol):
    """Protocol for abstracting file storage operations.

    All paths are **relative strings** (e.g., ``"eval/runs/step500/result.json"``).
    The storage backend resolves them against its root.
    """

    def read(self, path: str) -> bytes:
        """Read entire file. Raises ``FileNotFoundError`` if missing."""
        ...

    def write(self, path: str, data: bytes) -> None:
        """Write data to path, creating parent directories. Overwrites if exists."""
        ...

    def append(self, path: str, data: bytes) -> None:
        """Append data to file, creating it if needed. For JSONL streaming writes."""
        ...

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...

    def stat(self, path: str) -> StorageStat | None:
        """Get file size and mtime, or None if missing."""
        ...

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """Read bytes from offset. If length is None, read to end of file.

        Raises ``FileNotFoundError`` if missing.
        """
        ...

    def list_dir(self, prefix: str) -> list[str]:
        """List immediate children (files and directories) under prefix.

        Returns names only (not full paths). Empty list if prefix doesn't exist.
        """
        ...


class LocalStorage:
    """File-based storage rooted at a local directory.

    All paths are resolved relative to ``root``.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes storage root: {path}")
        return resolved

    def read(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        full = self._resolve(path)
        with open(full, "rb") as f:
            f.seek(offset)
            if length is not None:
                return f.read(length)
            return f.read()

    def write(self, path: str, data: bytes) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def append(self, path: str, data: bytes) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "ab") as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def stat(self, path: str) -> StorageStat | None:
        full = self._resolve(path)
        try:
            st = full.stat()
            return StorageStat(size=st.st_size, mtime=st.st_mtime)
        except FileNotFoundError:
            return None

    def list_dir(self, prefix: str) -> list[str]:
        full = self._resolve(prefix)
        if not full.is_dir():
            return []
        return sorted(child.name for child in full.iterdir())


def storage_join(*parts: str) -> str:
    """Join storage path segments, handling empty prefixes correctly."""
    return "/".join(p for p in parts if p)


# ── JSON/JSONL helpers ────────────────────────────────────────────────


def storage_read_json(storage: Storage, path: str) -> dict[str, Any] | None:
    """Read a JSON file from storage, returning None if missing or malformed."""
    try:
        data = storage.read(path)
        return json.loads(data)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read JSON %s: %s", path, e)
        return None


def storage_read_jsonl(storage: Storage, path: str) -> list[dict[str, Any]]:
    """Read a JSONL file from storage, returning an empty list if missing."""
    try:
        data = storage.read(path)
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning("Failed to read JSONL %s: %s", path, e)
        return []

    records: list[dict[str, Any]] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
    return records


def storage_write_json(storage: Storage, path: str, data: dict[str, Any]) -> None:
    """Write a dict as pretty-printed JSON to storage."""
    storage.write(path, json.dumps(data, indent=2).encode("utf-8"))


def storage_append_jsonl(storage: Storage, path: str, record: dict[str, Any]) -> None:
    """Append one JSON record as a line to a JSONL file in storage."""
    storage.append(path, (json.dumps(record) + "\n").encode("utf-8"))
