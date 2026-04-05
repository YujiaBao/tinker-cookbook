"""Shared I/O utilities for Tinker Chef data readers.

All file access goes through a ``Storage`` instance so the dashboard
can serve data from local disk, S3, or any other backend.
"""

import json
import logging
from typing import Any

from tinker_cookbook.storage import Storage

logger = logging.getLogger(__name__)

# Keep at most this many records in memory per reader.
# At 20 keys per record, 50K records ≈ 40MB — reasonable for a dashboard.
_MAX_RECORDS = 50_000


class IncrementalJsonlReader:
    """Base class for incremental JSONL file reading with offset tracking.

    Uses ``Storage.stat()`` to check for new data and ``Storage.read()``
    to fetch the full file contents. Tracks how many bytes have been
    parsed so only new lines are processed on each call.

    Records are kept in a bounded ring: once ``_MAX_RECORDS`` is reached,
    the oldest records are dropped to prevent unbounded memory growth.
    """

    def __init__(self, storage: Storage, path: str) -> None:
        self._storage = storage
        self._path = path
        self._offset: int = 0
        self._records: list[dict[str, Any]] = []
        self._total_read: int = 0  # total records ever read (for metrics)

    @property
    def path(self) -> str:
        return self._path

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    @property
    def total_read(self) -> int:
        """Total number of records read since creation (including dropped)."""
        return self._total_read

    def read(self) -> list[dict[str, Any]]:
        """Read new lines since last call. Returns only new records."""
        stat = self._storage.stat(self._path)
        if stat is None or stat.size <= self._offset:
            return []

        try:
            new_bytes = self._storage.read_range(self._path, self._offset)
        except FileNotFoundError:
            return []

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
        self._total_read += len(new_records)

        # Bound memory: drop oldest records if over limit
        if len(self._records) > _MAX_RECORDS:
            self._records = self._records[-_MAX_RECORDS:]

        return new_records

    def has_data(self) -> bool:
        if self._records:
            return True
        stat = self._storage.stat(self._path)
        return stat is not None and stat.size > 0
