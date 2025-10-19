from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .trainer import TinySpikingTrainer


class ResetRequest(BaseModel):
    seed: int | None = None


def create_app(trainer: TinySpikingTrainer | None = None) -> FastAPI:
    """Instantiate a FastAPI app that exposes interactive training endpoints."""
    trainer = trainer or TinySpikingTrainer()
    app = FastAPI(title="Tiny Spiking Mass Redistribution Playground")
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    app.state.trainer = trainer
    app.state.last_step: dict[str, Any] | None = None

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    index_path = static_dir / "index.html"

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/api/topology")
    async def topology() -> dict[str, Any]:
        return trainer.topology()

    @app.post("/api/reset")
    async def reset(request: ResetRequest) -> dict[str, Any]:
        trainer.reset(seed=request.seed)
        app.state.last_step = None
        return {"status": "reset", "seed": request.seed}

    @app.post("/api/step")
    async def step() -> dict[str, Any]:
        try:
            result = trainer.step()
        except Exception as exc:  # pragma: no cover - runtime guard for viz
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        app.state.last_step = result
        evaluation = trainer.evaluate()
        result["eval"] = evaluation
        return result

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        if app.state.last_step is None:
            return {"step": None}
        return app.state.last_step

    return app
