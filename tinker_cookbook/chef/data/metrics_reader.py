"""Incremental reader for metrics.jsonl files."""

from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.io import IncrementalJsonlReader


class MetricsReader(IncrementalJsonlReader):
    """Reads metrics.jsonl incrementally, tracking file offset between reads.

    Each line in metrics.jsonl is a JSON object with a ``step`` key and
    arbitrary metric key-value pairs.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._known_keys: set[str] = set()

    def read(self) -> list[dict[str, Any]]:
        new = super().read()
        for record in new:
            self._known_keys.update(k for k in record if k != "step")
        return new

    def metric_keys(self) -> set[str]:
        """Return the set of all metric keys seen across all records."""
        return set(self._known_keys)
