"""Reader for config.json files."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigReader:
    """Reads and caches a run's config.json.

    The config is immutable once written, so we read it once and cache.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._config: dict[str, Any] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> dict[str, Any] | None:
        """Read the config, returning the cached value on subsequent calls."""
        if self._config is not None:
            return self._config

        if not self._path.exists():
            return None

        try:
            with open(self._path) as f:
                self._config = json.load(f)
            return self._config
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read config %s: %s", self._path, e)
            return None
