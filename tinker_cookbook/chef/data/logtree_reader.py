"""Reader for logtree JSON exports."""

from typing import Any

from tinker_cookbook.storage import Storage, storage_join, storage_read_json


class LogtreeReader:
    """Reads logtree JSON files from iteration directories."""

    def __init__(self, storage: Storage, run_prefix: str) -> None:
        self._storage = storage
        self._prefix = run_prefix

    def read_logtree(
        self,
        iteration: int,
        base_name: str = "train",
    ) -> dict[str, Any] | None:
        path = storage_join(self._prefix, f"iteration_{iteration:06d}", f"{base_name}_logtree.json")
        return storage_read_json(self._storage, path)

    def list_logtrees(self, iteration: int) -> list[str]:
        iter_dir = storage_join(self._prefix, f"iteration_{iteration:06d}")
        items = self._storage.list_dir(iter_dir)
        return sorted(
            name[: -len("_logtree.json")]
            for name in items
            if name.endswith("_logtree.json")
        )
