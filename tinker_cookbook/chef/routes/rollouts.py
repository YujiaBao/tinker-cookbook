"""Rollout browser API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.data.store import RunStore

def create_router(store: RunStore) -> APIRouter:
    """Create the rollouts router bound to a specific RunStore."""
    router = APIRouter(prefix="/api/runs", tags=["rollouts"])

    @router.get("/{run_id}/iterations/{iteration}/rollouts")
    async def get_rollouts(
        run_id: str,
        iteration: int,
        split: str = Query("train", description="Dataset split (train or eval)"),
        label: str | None = Query(None, description="Eval label (for eval splits)"),
        tag: str | None = Query(None, description="Filter by tag"),
        min_reward: float | None = Query(None, description="Minimum total reward"),
        max_reward: float | None = Query(None, description="Maximum total reward"),
    ) -> dict[str, Any]:
        """Get rollout summaries for an iteration with optional filters."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        rollouts = store.get_rollouts(run_id, iteration, split, label)

        # Apply filters
        if tag is not None:
            rollouts = [r for r in rollouts if tag in r.get("tags", [])]
        if min_reward is not None:
            rollouts = [r for r in rollouts if r.get("total_reward", 0) >= min_reward]
        if max_reward is not None:
            rollouts = [r for r in rollouts if r.get("total_reward", 0) <= max_reward]

        # Build lightweight summaries (omit full step details for the list view)
        summaries = []
        for r in rollouts:
            summaries.append({
                "group_idx": r.get("group_idx"),
                "traj_idx": r.get("traj_idx"),
                "tags": r.get("tags", []),
                "total_reward": r.get("total_reward"),
                "final_reward": r.get("final_reward"),
                "num_steps": len(r.get("steps", [])),
                "final_ob_len": r.get("final_ob_len"),
                "sampling_client_step": r.get("sampling_client_step"),
            })

        # Collect all unique tags for filter UI
        all_tags: set[str] = set()
        all_rollouts = store.get_rollouts(run_id, iteration, split, label)
        for r in all_rollouts:
            all_tags.update(r.get("tags", []))

        return {
            "run_id": run_id,
            "iteration": iteration,
            "split": split,
            "total": len(summaries),
            "available_tags": sorted(all_tags),
            "rollouts": summaries,
        }

    @router.get("/{run_id}/iterations/{iteration}/rollouts/{group_idx}/{traj_idx}")
    async def get_rollout_detail(
        run_id: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        split: str = Query("train"),
        label: str | None = Query(None),
    ) -> dict[str, Any]:
        """Get full detail for a single rollout including all steps."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        rollout = store.get_single_rollout(run_id, iteration, group_idx, traj_idx, split, label)
        if rollout is None:
            raise HTTPException(
                status_code=404,
                detail=f"Rollout ({group_idx}, {traj_idx}) not found at iteration {iteration}",
            )
        return rollout

    @router.get("/{run_id}/iterations/{iteration}/logtree")
    async def get_logtree(
        run_id: str,
        iteration: int,
        base_name: str = Query("train"),
    ) -> dict[str, Any]:
        """Get the logtree JSON for an iteration."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        logtree = store.get_logtree(run_id, iteration, base_name)
        if logtree is None:
            raise HTTPException(
                status_code=404,
                detail=f"No logtree found for iteration {iteration}, base_name={base_name}",
            )
        return logtree

    return router
