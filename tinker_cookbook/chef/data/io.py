"""Shared I/O utilities for Tinker Chef data readers.

All file access goes through a ``Storage`` instance so the dashboard
can serve data from local disk, S3, or any other backend.
"""

import json
import logging
from typing import Any

from tinker_cookbook.storage import Storage, storage_read_json, storage_read_jsonl

logger = logging.getLogger(__name__)

# Re-export for backward compat with readers that import from here
read_json_from_storage = storage_read_json
read_jsonl_from_storage = storage_read_jsonl


class IncrementalJsonlReader:
    """Base class for incremental JSONL file reading with offset tracking.

    Uses ``Storage.stat()`` to check for new data and ``Storage.read()``
    to fetch the full file contents. Tracks how many bytes have been
    parsed so only new lines are processed on each call.
    """

    def __init__(self, storage: Storage, path: str) -> None:
        self._storage = storage
        self._path = path
        self._offset: int = 0
        self._records: list[dict[str, Any]] = []

    @property
    def path(self) -> str:
        return self._path

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def read(self) -> list[dict[str, Any]]:
        """Read new lines since last call. Returns only new records."""
        stat = self._storage.stat(self._path)
        if stat is None or stat.size <= self._offset:
            return []

        try:
            raw_bytes = self._storage.read(self._path)
        except FileNotFoundError:
            return []

        # Skip already-parsed bytes
        new_bytes = raw_bytes[self._offset:]

        # Only process complete lines
        if not new_bytes.endswith(b"\n"):
            last_newline = new_bytes.rfind(b"\n")
            if last_newline == -1:
                return []
            new_bytes = new_bytes[: last_newline + 1]

        self._offset += len(new_bytes)
        raw = new_bytes.decode("utf-8", errors="replace")

        new_records: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                new_records.append(record)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line: %s", line[:100])

        self._records.extend(new_records)
        return new_records

    def has_data(self) -> bool:
        if self._records:
            return True
        stat = self._storage.stat(self._path)
        return stat is not None and stat.size > 0
