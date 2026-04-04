"""Eval benchmark API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.data.store import RunStore


def create_router(store: RunStore) -> APIRouter:
    """Create the eval router bound to a specific RunStore."""
    router = APIRouter(prefix="/api/eval", tags=["eval"])

    @router.get("/runs")
    async def list_eval_runs() -> list[dict[str, Any]]:
        """List all eval runs."""
        reader = store.get_global_eval_reader()
        if reader is None:
            return []
        runs = reader.list_eval_runs()
        result = []
        for run_entry in runs:
            run_id = run_entry.get("run_id", "")
            if not run_id:
                continue
            metadata = reader.get_eval_run_metadata(run_id)
            benchmarks = reader.list_benchmarks(run_id)
            result.append({
                "eval_run_id": run_id,
                "model_name": metadata.get("model_name", "") if metadata else "",
                "checkpoint_path": metadata.get("checkpoint_path") if metadata else None,
                "checkpoint_name": metadata.get("checkpoint_name") if metadata else None,
                "timestamp": metadata.get("timestamp") if metadata else None,
                "benchmarks": benchmarks,
                "scores": metadata.get("scores", {}) if metadata else {},
            })
        return result

    @router.get("/runs/{eval_run_id}")
    async def get_eval_run(eval_run_id: str) -> dict[str, Any]:
        """Get metadata and results for a specific eval run."""
        reader = store.get_global_eval_reader()
        if reader is None:
            raise HTTPException(status_code=404, detail="No eval data found")
        metadata = reader.get_eval_run_metadata(eval_run_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Eval run '{eval_run_id}' not found")

        benchmarks = reader.list_benchmarks(eval_run_id)
        results: dict[str, Any] = {}
        for benchmark in benchmarks:
            result = reader.get_benchmark_result(eval_run_id, benchmark)
            if result:
                results[benchmark] = result

        return {
            "eval_run_id": eval_run_id,
            "metadata": metadata,
            "benchmarks": benchmarks,
            "results": results,
        }

    @router.get("/runs/{eval_run_id}/{benchmark}/trajectories")
    async def get_eval_trajectories(
        eval_run_id: str,
        benchmark: str,
        correct_only: bool = Query(False),
        errors_only: bool = Query(False),
    ) -> dict[str, Any]:
        """Get eval trajectories with optional filters."""
        reader = store.get_global_eval_reader()
        if reader is None:
            raise HTTPException(status_code=404, detail="No eval data found")

        trajectories = reader.get_benchmark_trajectories(eval_run_id, benchmark)

        if correct_only:
            trajectories = [t for t in trajectories if t.get("reward", 0) > 0]
        if errors_only:
            trajectories = [t for t in trajectories if t.get("error") is not None]

        summaries = []
        for t in trajectories:
            summaries.append({
                "idx": t.get("idx"),
                "example_id": t.get("example_id"),
                "reward": t.get("reward", 0),
                "num_turns": len(t.get("turns", [])),
                "time_seconds": t.get("time_seconds", 0),
                "error": t.get("error"),
                "logs": t.get("logs", {}),
            })

        return {
            "eval_run_id": eval_run_id,
            "benchmark": benchmark,
            "total": len(summaries),
            "trajectories": summaries,
        }

    @router.get("/runs/{eval_run_id}/{benchmark}/trajectories/{idx}")
    async def get_eval_trajectory_detail(
        eval_run_id: str,
        benchmark: str,
        idx: int,
    ) -> dict[str, Any]:
        """Get full trajectory detail including all turns."""
        reader = store.get_global_eval_reader()
        if reader is None:
            raise HTTPException(status_code=404, detail="No eval data found")

        traj = reader.get_single_trajectory(eval_run_id, benchmark, idx)
        if traj is None:
            raise HTTPException(
                status_code=404,
                detail=f"Trajectory {idx} not found in {benchmark}",
            )
        return traj

    @router.get("/scores")
    async def get_scores_table() -> list[dict[str, Any]]:
        """Get a scores table for the model progression view."""
        reader = store.get_global_eval_reader()
        if reader is None:
            return []
        return reader.get_scores_table()

    return router
