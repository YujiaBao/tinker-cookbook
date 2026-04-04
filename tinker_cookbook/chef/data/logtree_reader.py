"""Reader for logtree JSON exports."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LogtreeReader:
    """Reads logtree JSON files from iteration directories.

    Logtree files contain structured HTML tree data with nested nodes
    that include conversation data, metrics, and diagnostic info.
    """

    def __init__(self, run_path: Path) -> None:
        self._run_path = run_path

    def read_logtree(
        self,
        iteration: int,
        base_name: str = "train",
    ) -> dict[str, Any] | None:
        """Read a logtree JSON for a specific iteration.

        Args:
            iteration: The iteration number.
            base_name: File prefix ("train" or "eval_LABEL").

        Returns:
            Parsed logtree JSON, or None if not found.
        """
        iter_dir = self._run_path / f"iteration_{iteration:06d}"
        path = iter_dir / f"{base_name}_logtree.json"
        return self._read_json_cached(path)

    def list_logtrees(self, iteration: int) -> list[str]:
        """List available logtree base names for an iteration."""
        iter_dir = self._run_path / f"iteration_{iteration:06d}"
        if not iter_dir.is_dir():
            return []

        names: list[str] = []
        for f in iter_dir.iterdir():
            if f.name.endswith("_logtree.json"):
                base = f.name[: -len("_logtree.json")]
                names.append(base)
        return sorted(names)

    @staticmethod
    @lru_cache(maxsize=16)
    def _read_json_cached(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read logtree %s: %s", path, e)
            return None
