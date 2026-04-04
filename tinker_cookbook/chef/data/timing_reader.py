"""Incremental reader for timing_spans.jsonl files."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TimingReader:
    """Reads timing_spans.jsonl incrementally with offset tracking.

    Each line contains span records for a training step, with fields:
    name, start_time, end_time, wall_start, wall_end, step.
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
        if not self._path.exists():
            return []

        file_size = self._path.stat().st_size
        if file_size <= self._offset:
            return []

        new_records: list[dict[str, Any]] = []
        with open(self._path, "r") as f:
            f.seek(self._offset)
            raw = f.read()

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
                logger.warning("Skipping malformed timing line: %s", line[:100])

        self._records.extend(new_records)
        return new_records

    def get_spans_for_step(self, step: int) -> list[dict[str, Any]]:
        """Return all span records for a given training step."""
        return [r for r in self._records if r.get("step") == step]
