"""Reader for config.json files."""

from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.io import read_json


class ConfigReader:
    """Reads and caches a run's config.json.

    The config is immutable once written, so we read it once and cache.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._config: dict[str, Any] | None = None
        self._read_attempted: bool = False

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> dict[str, Any] | None:
        """Read the config, returning the cached value on subsequent calls."""
        if self._read_attempted:
            return self._config
        self._read_attempted = True
        self._config = read_json(self._path)
        return self._config
