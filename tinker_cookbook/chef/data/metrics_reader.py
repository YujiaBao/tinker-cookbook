"""Incremental reader for metrics.jsonl files.

Tracks the file byte offset so subsequent reads only parse new lines,
making it efficient for live-updating dashboards during training.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsReader:
    """Reads metrics.jsonl incrementally, tracking file offset between reads.

    Each line in metrics.jsonl is a JSON object with a ``step`` key and
    arbitrary metric key-value pairs.  The reader keeps all parsed records
    in memory and only reads new bytes on each call to :meth:`read`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset: int = 0
        self._records: list[dict[str, Any]] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def records(self) -> list[dict[str, Any]]:
        """All records read so far (oldest first)."""
        return self._records

    def read(self) -> list[dict[str, Any]]:
        """Read any new lines appended since the last call.

        Returns only the *new* records.  The full history is available
        via :attr:`records`.
        """
        if not self._path.exists():
            return []

        file_size = self._path.stat().st_size
        if file_size <= self._offset:
            return []

        new_records: list[dict[str, Any]] = []
        with open(self._path, "r") as f:
            f.seek(self._offset)
            raw = f.read()

        # Only process complete lines (ignore a trailing partial write)
        if not raw.endswith("\n"):
            last_newline = raw.rfind("\n")
            if last_newline == -1:
                return []
            raw = raw[: last_newline + 1]

        self._offset += len(raw.encode("utf-8"))

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                new_records.append(record)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed metrics line: %s", line[:100])

        self._records.extend(new_records)
        return new_records

    def has_data(self) -> bool:
        """True if at least one record has been read or the file exists with content."""
        if self._records:
            return True
        return self._path.exists() and self._path.stat().st_size > 0

    def metric_keys(self) -> set[str]:
        """Return the set of all metric keys seen across all records."""
        keys: set[str] = set()
        for record in self._records:
            keys.update(record.keys())
        keys.discard("step")
        return keys
