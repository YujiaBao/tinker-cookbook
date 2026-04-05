"""Incremental reader for metrics.jsonl files."""

from typing import Any

from tinker_cookbook.chef.data.io import IncrementalJsonlReader
from tinker_cookbook.storage import Storage


class MetricsReader(IncrementalJsonlReader):
    """Reads metrics.jsonl incrementally, tracking file offset between reads."""

    def __init__(self, storage: Storage, path: str) -> None:
        super().__init__(storage, path)
        self._known_keys: set[str] = set()

    def read(self) -> list[dict[str, Any]]:
        new = super().read()
        for record in new:
            self._known_keys.update(k for k in record if k != "step")
        return new

    def metric_keys(self) -> set[str]:
        return set(self._known_keys)
