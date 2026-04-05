"""Metrics API routes including SSE streaming."""

import asyncio
import json
import logging
from fnmatch import fnmatch
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from tinker_cookbook.chef.data.store import RunStore

logger = logging.getLogger(__name__)

def create_router(store: RunStore) -> APIRouter:
    """Create the metrics router bound to a specific RunStore."""
    router = APIRouter(prefix="/api/runs", tags=["metrics"])

    @router.get("/{run_id}/metrics")
    async def get_metrics(
        run_id: str,
        keys: str | None = Query(None, description="Comma-separated glob patterns for metric keys"),
    ) -> dict[str, Any]:
        """Get all metrics for a run, optionally filtered by key patterns."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        records = store.get_metrics(run_id)

        if keys:
            patterns = [p.strip() for p in keys.split(",")]
            records = [_filter_record(r, patterns) for r in records]

        return {
            "run_id": run_id,
            "total_records": len(records),
            "records": records,
        }

    @router.get("/{run_id}/metrics/keys")
    async def get_metric_keys(run_id: str) -> list[str]:
        """Get the set of all metric keys for a run."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        reader = store.get_metrics_reader(run_id)
        if reader is None:
            return []
        reader.read()
        return sorted(reader.metric_keys())

    @router.get("/{run_id}/metrics/stream")
    async def stream_metrics(run_id: str, request: Request) -> StreamingResponse:
        """SSE endpoint that streams new metric lines as they appear."""
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        async def event_generator():
            store.get_metrics(run_id)
            idle_cycles = 0
            max_idle_cycles = 40  # 40 * 15s = 10 minutes of no new data

            while True:
                if await request.is_disconnected():
                    break

                new_records = store.get_new_metrics(run_id)
                if new_records:
                    idle_cycles = 0
                    for record in new_records:
                        data = json.dumps(record)
                        yield f"data: {data}\n\n"
                else:
                    idle_cycles += 1
                    yield ": keepalive\n\n"
                    if idle_cycles >= max_idle_cycles:
                        yield "event: timeout\ndata: {}\n\n"
                        break

                await asyncio.sleep(15)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router


def _filter_record(record: dict[str, Any], patterns: list[str]) -> dict[str, Any]:
    """Filter a metrics record to only include keys matching glob patterns.

    The ``step`` key is always included.
    """
    filtered: dict[str, Any] = {}
    for key, value in record.items():
        if key == "step":
            filtered[key] = value
            continue
        if any(fnmatch(key, pattern) for pattern in patterns):
            filtered[key] = value
    return filtered
