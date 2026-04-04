"""FastAPI application factory for Tinker Chef."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tinker_cookbook.chef.data.store import RunStore
from tinker_cookbook.chef.routes import metrics, rollouts, runs, timing

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(root: str | Path) -> FastAPI:
    """Create and configure the Tinker Chef FastAPI application.

    Args:
        root: Root directory to scan for training runs. Can be a single
            run directory or a parent directory containing multiple runs.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root_path}")

    store = RunStore(root_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        run_list = store.refresh_runs()
        logger.info("Tinker Chef started — discovered %d run(s) in %s", len(run_list), root_path)
        for run in run_list:
            logger.info("  Run '%s': %d iterations", run.run_id, run.iteration_count)
        yield

    app = FastAPI(
        title="Tinker Chef",
        description="Training visualization dashboard for tinker-cookbook",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development (React dev server on different port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(runs.create_router(store))
    app.include_router(metrics.create_router(store))
    app.include_router(rollouts.create_router(store))
    app.include_router(timing.create_router(store))

    # Serve pre-built React static files if they exist
    if (_STATIC_DIR / "index.html").exists():
        # Mount static assets (js, css, etc.)
        assets_dir = _STATIC_DIR / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # SPA fallback: serve index.html for all non-API routes
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            static_file = _STATIC_DIR / full_path
            if static_file.is_file() and not full_path.startswith("api/"):
                return FileResponse(str(static_file))
            return FileResponse(str(_STATIC_DIR / "index.html"))

    else:
        # No frontend built yet — show a helpful message
        @app.get("/")
        async def no_frontend() -> dict[str, str]:
            return {
                "message": "Tinker Chef API is running. Frontend not built yet.",
                "hint": "Run 'cd tinker_cookbook/chef/frontend && npm install && npm run build' to build the frontend.",
                "api_docs": "/docs",
            }

    return app
