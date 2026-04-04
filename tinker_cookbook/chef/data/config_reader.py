"""Reader for config.json files."""

from typing import Any

from tinker_cookbook.storage import Storage, storage_read_json


class ConfigReader:
    """Reads and caches a run's config.json."""

    def __init__(self, storage: Storage, path: str) -> None:
        self._storage = storage
        self._path = path
        self._config: dict[str, Any] | None = None
        self._read_attempted: bool = False

    def read(self) -> dict[str, Any] | None:
        if self._read_attempted:
            return self._config
        self._read_attempted = True
        self._config = storage_read_json(self._storage, self._path)
        return self._config
