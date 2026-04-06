"""Storage protocols and LocalStorage implementation.

Pure byte-level I/O — no JSON, no application logic. The Storage protocol
is the extension point for adding S3, GCS, or other backends.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageStat:
    """File metadata."""

    size: int
    mtime: float  # seconds since epoch


@runtime_checkable
class Storage(Protocol):
    """Sync byte-level file I/O.

    All paths are relative strings (e.g., ``"runs/001/metrics.jsonl"``).
    The backend resolves them against its root.
    """

    def read(self, path: str) -> bytes:
        """Read entire file. Raises ``FileNotFoundError`` if missing."""
        ...

    def write(self, path: str, data: bytes) -> None:
        """Write data, creating parent dirs. Overwrites if exists."""
        ...

    def append(self, path: str, data: bytes) -> None:
        """Append data to file, creating if needed.

        Atomic for writes < PIPE_BUF (~4KB) on POSIX.
        Cloud backends may raise ``NotImplementedError`` — stores
        handle the shard pattern at a higher level.
        """
        ...

    def exists(self, path: str) -> bool:
        ...

    def stat(self, path: str) -> StorageStat | None:
        """Get file size and mtime, or ``None`` if missing."""
        ...

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """Read bytes from offset. If length is None, read to end.

        Raises ``FileNotFoundError`` if missing.
        """
        ...

    def list_dir(self, prefix: str) -> list[str]:
        """List immediate children under prefix. Returns names only."""
        ...

    def remove(self, path: str) -> None:
        """Delete a file. No error if missing."""
        ...

    def __enter__(self) -> Storage:
        ...

    def __exit__(self, *exc: object) -> None:
        ...


@runtime_checkable
class AsyncStorage(Protocol):
    """Async byte-level file I/O.

    Cloud backends (S3, GCS) implement this natively. LocalStorage
    wraps sync methods with ``asyncio.to_thread``.
    """

    async def aread(self, path: str) -> bytes:
        ...

    async def awrite(self, path: str, data: bytes) -> None:
        ...

    async def aappend(self, path: str, data: bytes) -> None:
        ...

    async def aexists(self, path: str) -> bool:
        ...

    async def astat(self, path: str) -> StorageStat | None:
        ...

    async def aread_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        ...

    async def alist_dir(self, prefix: str) -> list[str]:
        ...

    async def aremove(self, path: str) -> None:
        ...

    async def __aenter__(self) -> AsyncStorage:
        ...

    async def __aexit__(self, *exc: object) -> None:
        ...


class LocalStorage:
    """File-based storage rooted at a local directory.

    Implements both ``Storage`` (sync) and ``AsyncStorage`` (via ``to_thread``).
    Pickle-serializable (stores only a ``Path``).
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

    # --- Sync (Storage protocol) ---

    def read(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

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
        try:
            st = self._resolve(path).stat()
            return StorageStat(size=st.st_size, mtime=st.st_mtime)
        except FileNotFoundError:
            return None

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        with open(self._resolve(path), "rb") as f:
            f.seek(offset)
            return f.read(length) if length is not None else f.read()

    def list_dir(self, prefix: str) -> list[str]:
        full = self._resolve(prefix)
        if not full.is_dir():
            return []
        return sorted(child.name for child in full.iterdir())

    def remove(self, path: str) -> None:
        full = self._resolve(path)
        full.unlink(missing_ok=True)

    def __enter__(self) -> LocalStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    # --- Async (AsyncStorage protocol, via to_thread) ---

    async def aread(self, path: str) -> bytes:
        return await asyncio.to_thread(self.read, path)

    async def awrite(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(self.write, path, data)

    async def aappend(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(self.append, path, data)

    async def aexists(self, path: str) -> bool:
        return await asyncio.to_thread(self.exists, path)

    async def astat(self, path: str) -> StorageStat | None:
        return await asyncio.to_thread(self.stat, path)

    async def aread_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        return await asyncio.to_thread(self.read_range, path, offset, length)

    async def alist_dir(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(self.list_dir, prefix)

    async def aremove(self, path: str) -> None:
        await asyncio.to_thread(self.remove, path)

    async def __aenter__(self) -> LocalStorage:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


def storage_join(*parts: str) -> str:
    """Join storage path segments, filtering empty strings."""
    return "/".join(p for p in parts if p)
