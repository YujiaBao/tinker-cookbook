"""Timing data API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.data.store import RunStore

def create_router(store: RunStore) -> APIRouter:
    """Create the timing router bound to a specific RunStore."""
    router = APIRouter(prefix="/api/runs", tags=["timing"])

    @router.get("/{run_id}/timing")
    async def get_timing(
        run_id: str,
        step_start: int | None = Query(None, description="Start step (inclusive)"),
        step_end: int | None = Query(None, description="End step (inclusive)"),
    ) -> dict[str, Any]:
        """Get timing span data, optionally filtered by step range."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        records = store.get_timing(run_id)

        if step_start is not None:
            records = [r for r in records if r.get("step", 0) >= step_start]
        if step_end is not None:
            records = [r for r in records if r.get("step", 0) <= step_end]

        return {
            "run_id": run_id,
            "total_records": len(records),
            "records": records,
        }

    return router
