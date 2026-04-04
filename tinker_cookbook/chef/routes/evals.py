"""Eval benchmark API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.data.eval_reader import EvalReader

router = APIRouter(prefix="/api/evals", tags=["evals"])


def create_router(eval_reader: EvalReader | None) -> APIRouter:
    """Create the evals router bound to an EvalReader (may be None if no eval data)."""

    @router.get("/runs")
    async def list_eval_runs() -> list[dict[str, Any]]:
        """List all eval runs."""
        if eval_reader is None:
            return []
        return eval_reader.list_eval_runs()

    @router.get("/runs/{run_id}")
    async def get_eval_run(run_id: str) -> dict[str, Any]:
        """Get metadata for an eval run."""
        if eval_reader is None:
            raise HTTPException(status_code=404, detail="No eval data available")
        metadata = eval_reader.get_eval_run_metadata(run_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found")
        metadata["benchmarks"] = eval_reader.list_benchmarks(run_id)
        return metadata

    @router.get("/runs/{run_id}/benchmarks/{benchmark}/result")
    async def get_benchmark_result(run_id: str, benchmark: str) -> dict[str, Any]:
        """Get aggregated result for a benchmark."""
        if eval_reader is None:
            raise HTTPException(status_code=404, detail="No eval data available")
        result = eval_reader.get_benchmark_result(run_id, benchmark)
        if result is None:
            raise HTTPException(status_code=404, detail="Benchmark result not found")
        return result

    @router.get("/runs/{run_id}/benchmarks/{benchmark}/trajectories")
    async def get_benchmark_trajectories(
        run_id: str,
        benchmark: str,
        correct_only: bool = Query(False),
        errors_only: bool = Query(False),
    ) -> dict[str, Any]:
        """Get stored trajectories for a benchmark with optional filters."""
        if eval_reader is None:
            raise HTTPException(status_code=404, detail="No eval data available")
        trajectories = eval_reader.get_benchmark_trajectories(run_id, benchmark)

        if correct_only:
            trajectories = [t for t in trajectories if t.get("reward", 0) > 0]
        if errors_only:
            trajectories = [t for t in trajectories if t.get("error") is not None]

        # Return lightweight summaries
        summaries = []
        for t in trajectories:
            summaries.append({
                "idx": t.get("idx"),
                "example_id": t.get("example_id"),
                "reward": t.get("reward"),
                "error": t.get("error"),
                "num_turns": len(t.get("turns", [])),
                "time_seconds": t.get("time_seconds"),
                "benchmark": t.get("benchmark"),
            })

        return {
            "run_id": run_id,
            "benchmark": benchmark,
            "total": len(summaries),
            "trajectories": summaries,
        }

    @router.get("/runs/{run_id}/benchmarks/{benchmark}/trajectories/{idx}")
    async def get_single_trajectory(run_id: str, benchmark: str, idx: int) -> dict[str, Any]:
        """Get full detail for a single eval trajectory."""
        if eval_reader is None:
            raise HTTPException(status_code=404, detail="No eval data available")
        traj = eval_reader.get_single_trajectory(run_id, benchmark, idx)
        if traj is None:
            raise HTTPException(status_code=404, detail=f"Trajectory {idx} not found")
        return traj

    @router.get("/scores")
    async def get_scores_table() -> list[dict[str, Any]]:
        """Get a scores table across all eval runs and benchmarks.

        Returns rows of {run_id, model_name, checkpoint_name, scores: {benchmark: score}}.
        """
        if eval_reader is None:
            return []
        return eval_reader.get_scores_table()

    return router
