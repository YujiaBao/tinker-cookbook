"""Timing data API routes with concurrency analysis."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.data.store import RunStore

router = APIRouter(prefix="/api/runs", tags=["timing"])


def create_router(store: RunStore) -> APIRouter:
    """Create the timing router bound to a specific RunStore."""

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

    @router.get("/{run_id}/timing/flat")
    async def get_timing_flat(
        run_id: str,
        step_start: int | None = Query(None),
        step_end: int | None = Query(None),
    ) -> dict[str, Any]:
        """Get all spans flattened into individual records with step annotation.

        Each span has: step, name, duration, wall_start, wall_end.
        """
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        reader = store.get_timing_reader(run_id)
        if reader is None:
            return {"run_id": run_id, "spans": []}
        reader.read()

        spans = reader.get_all_spans_flat()
        if step_start is not None:
            spans = [s for s in spans if s.get("step", 0) >= step_start]
        if step_end is not None:
            spans = [s for s in spans if s.get("step", 0) <= step_end]

        return {
            "run_id": run_id,
            "total_spans": len(spans),
            "spans": spans,
        }

    @router.get("/{run_id}/timing/concurrency/{step}")
    async def get_concurrency(run_id: str, step: int) -> dict[str, Any]:
        """Analyze concurrency for a specific training step.

        Returns span data with overlap analysis showing which operations
        ran in parallel. Visualizes Tinker SDK's async Future execution model.

        Response includes:
        - ``spans``: Individual spans sorted by start time
        - ``max_concurrency``: Peak number of overlapping operations
        - ``timeline``: Time series of concurrency level changes
        """
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        reader = store.get_timing_reader(run_id)
        if reader is None:
            return {"step": step, "spans": [], "max_concurrency": 0, "timeline": []}
        reader.read()

        return reader.get_concurrency_analysis(step)

    return router
