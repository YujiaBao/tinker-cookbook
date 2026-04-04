"""Reader for logtree JSON exports."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.io import read_json


class LogtreeReader:
    """Reads logtree JSON files from iteration directories."""

    def __init__(self, run_path: Path) -> None:
        self._run_path = run_path

    def read_logtree(
        self,
        iteration: int,
        base_name: str = "train",
    ) -> dict[str, Any] | None:
        """Read a logtree JSON for a specific iteration."""
        iter_dir = self._run_path / f"iteration_{iteration:06d}"
        return self._read_cached(iter_dir / f"{base_name}_logtree.json")

    def list_logtrees(self, iteration: int) -> list[str]:
        """List available logtree base names for an iteration."""
        iter_dir = self._run_path / f"iteration_{iteration:06d}"
        if not iter_dir.is_dir():
            return []
        return sorted(
            f.name[: -len("_logtree.json")]
            for f in iter_dir.iterdir()
            if f.name.endswith("_logtree.json")
        )

    @staticmethod
    @lru_cache(maxsize=16)
    def _read_cached(path: Path) -> dict[str, Any] | None:
        return read_json(path)
