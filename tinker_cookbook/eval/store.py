"""Backward-compatible re-export of EvalStore from stores/.

The canonical implementation now lives in ``tinker_cookbook.stores.eval_store``.
This module re-exports everything so existing imports continue to work.

For new code, import directly from ``tinker_cookbook.stores``.
"""

from tinker_cookbook.stores.eval_store import EvalStore, RunComparison, RunMetadata

__all__ = ["EvalStore", "RunComparison", "RunMetadata"]
