"""Run discovery and detail API routes."""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from tinker_cookbook.chef.data.store import RunStore

def create_router(store: RunStore) -> APIRouter:
    """Create the runs router bound to a specific RunStore."""
    router = APIRouter(prefix="/api/runs", tags=["runs"])

    @router.get("")
    async def list_runs() -> list[dict[str, Any]]:
        """List all discovered training runs."""
        runs = store.get_runs()
        result = []
        for run in runs:
            info = asdict(run)
            # Don't serialize the full Path object
            info["path"] = str(run.path)
            # Include a config summary if available
            config = store.get_config(run.run_id)
            if config:
                info["config_summary"] = _extract_config_summary(config)
            # Include latest step count
            metrics = store.get_metrics(run.run_id)
            if metrics:
                last = metrics[-1]
                info["latest_step"] = last.get("step")
            result.append(info)
        return result

    @router.get("/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        """Get details for a single run."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        info = asdict(run)
        info["path"] = str(run.path)
        config = store.get_config(run_id)
        if config:
            info["config"] = config
        metrics = store.get_metrics(run_id)
        if metrics:
            info["latest_step"] = metrics[-1].get("step")
            info["total_steps"] = len(metrics)
        return info

    @router.get("/{run_id}/config")
    async def get_config(run_id: str) -> dict[str, Any]:
        """Get the full config for a run."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        config = store.get_config(run_id)
        if config is None:
            raise HTTPException(status_code=404, detail="No config.json found")
        return config

    @router.get("/{run_id}/iterations")
    async def list_iterations(run_id: str) -> list[dict[str, Any]]:
        """List available iterations for a run."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        iterations = store.get_iterations(run_id)
        return [
            {
                "iteration": it.iteration,
                "has_train_rollouts": it.has_train_rollouts,
                "has_train_logtree": it.has_train_logtree,
                "eval_labels": it.eval_labels,
            }
            for it in iterations
        ]

    @router.get("/{run_id}/checkpoints")
    async def get_checkpoints(run_id: str) -> list[dict[str, Any]]:
        """Get checkpoint records for a run."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        # Reuse metrics reader pattern for checkpoints.jsonl
        import json
        from pathlib import Path

        ckpt_path = Path(run.path) / "checkpoints.jsonl"
        if not ckpt_path.exists():
            return []
        records = []
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    return router


def _extract_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from a config for display in the run list."""
    summary: dict[str, Any] = {}
    for key in ("model_name", "learning_rate", "batch_size", "n_batches", "lora_rank"):
        if key in config:
            summary[key] = config[key]
    return summary
