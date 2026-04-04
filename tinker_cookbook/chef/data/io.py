"""Shared I/O utilities for Tinker Chef data readers."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning None if missing or malformed."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read JSON %s: %s", path, e)
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, returning an empty list if missing or malformed."""
    records: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read JSONL %s: %s", path, e)
    return records


class IncrementalJsonlReader:
    """Base class for incremental JSONL file reading with byte-offset tracking.

    Reads only new bytes since the last call to :meth:`read`, making it
    efficient for files that are appended to during training.
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
        return self._records

    def read(self) -> list[dict[str, Any]]:
        """Read new lines since last call. Returns only new records."""
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return []

        if size <= self._offset:
            return []

        new_records: list[dict[str, Any]] = []
        with open(self._path, "rb") as f:
            f.seek(self._offset)
            raw_bytes = f.read()

        # Only process complete lines (ignore trailing partial write)
        if not raw_bytes.endswith(b"\n"):
            last_newline = raw_bytes.rfind(b"\n")
            if last_newline == -1:
                return []
            raw_bytes = raw_bytes[: last_newline + 1]

        self._offset += len(raw_bytes)
        raw = raw_bytes.decode("utf-8", errors="replace")

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
        try:
            return self._path.stat().st_size > 0
        except FileNotFoundError:
            return False
