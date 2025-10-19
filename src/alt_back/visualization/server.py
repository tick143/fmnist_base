from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .trainer import (
    TinySpikingTrainer,
    TrainerConfig,
    trainer_config_from_dict,
    trainer_config_to_dict,
    trainer_config_from_yaml,
)


class ResetRequest(BaseModel):
    seed: int | None = None


def _instantiate_trainer(config: TrainerConfig | None = None) -> TinySpikingTrainer:
    return TinySpikingTrainer(config=config)


def create_app(
    trainer: TinySpikingTrainer | None = None,
    config_path: str | None = None,
) -> FastAPI:
    """Instantiate a FastAPI app that exposes interactive training endpoints."""
    if trainer is None:
        config = trainer_config_from_yaml(config_path) if config_path else None
        trainer = _instantiate_trainer(config=config)

    app = FastAPI(title="Tiny Spiking Mass Redistribution Playground")
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    app.state.trainer = trainer
    app.state.last_step: dict[str, Any] | None = None
    app.state.config_path = config_path

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    index_path = static_dir / "index.html"

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/api/topology")
    async def topology() -> dict[str, Any]:
        return app.state.trainer.topology()

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return {
            "config": trainer_config_to_dict(app.state.trainer.config),
            "config_path": app.state.config_path,
        }

    @app.post("/api/reset")
    async def reset(request: ResetRequest) -> dict[str, Any]:
        app.state.trainer.reset(seed=request.seed)
        app.state.last_step = None
        return {"status": "reset", "seed": request.seed}

    @app.post("/api/reconfigure")
    async def reconfigure(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            new_config = trainer_config_from_dict(payload, base=app.state.trainer.config)
            app.state.trainer = _instantiate_trainer(config=new_config)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        app.state.last_step = None
        updated_config = trainer_config_to_dict(app.state.trainer.config)
        return {
            "status": "reconfigured",
            "config": updated_config,
            "topology": app.state.trainer.topology(),
        }

    @app.post("/api/reload")
    async def reload_config() -> dict[str, Any]:
        if app.state.config_path is None:
            raise HTTPException(status_code=400, detail="Server was not started with a config file.")
        try:
            cfg = trainer_config_from_yaml(app.state.config_path, base=app.state.trainer.config)
            app.state.trainer = _instantiate_trainer(config=cfg)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - config parse guard
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        app.state.last_step = None
        updated_config = trainer_config_to_dict(app.state.trainer.config)
        return {
            "status": "reloaded",
            "config": updated_config,
            "topology": app.state.trainer.topology(),
        }

    @app.post("/api/step")
    async def step() -> dict[str, Any]:
        try:
            result = app.state.trainer.step()
        except Exception as exc:  # pragma: no cover - runtime guard for viz
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        app.state.last_step = result
        return result

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        if app.state.last_step is None:
            return {"step": None}
        return app.state.last_step

    return app
