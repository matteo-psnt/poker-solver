"""FastAPI app factory and local server wrapper for chart viewing and play."""

from __future__ import annotations

import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.api.chart_service import ChartService
from src.interfaces.api.models import (
    ActionRequest,
    ChartResponse,
    HealthResponse,
    MetaResponse,
    NewHandRequest,
)
from src.interfaces.api.play_service import PlayService


class ChartServiceProtocol(Protocol):
    def get_metadata(self) -> dict: ...

    def get_chart(self, position: int, situation_id: str) -> dict: ...


def _resolve_static_path(static_dir: Path, requested_path: str) -> Path | None:
    candidate = (static_dir / requested_path).resolve()
    try:
        candidate.relative_to(static_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() and candidate.is_file() else None


def _register_chart_routes(app: FastAPI, service: ChartServiceProtocol) -> None:
    @app.get("/api/meta", response_model=MetaResponse)
    def meta() -> dict:
        return service.get_metadata()

    @app.get("/api/chart", response_model=ChartResponse)
    def chart(
        position: int = Query(default=0, ge=0, le=1),
        situation: str = Query(default="first_to_act", min_length=1),
    ) -> dict:
        try:
            return service.get_chart(position, situation)
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc


def _register_play_routes(app: FastAPI, service: PlayService) -> None:
    @app.get("/api/game/meta")
    def game_meta() -> dict:
        return {"runId": service.run_id, "infosets": service.num_infosets()}

    @app.post("/api/game/new")
    def new_hand(request: NewHandRequest) -> dict:
        return service.new_hand(human_seat=request.human_seat, button=request.button)

    @app.get("/api/game/{session_id}")
    def game_state(session_id: str) -> dict:
        try:
            return service.get_state(session_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND, detail="Unknown or expired session"
            ) from exc

    @app.post("/api/game/{session_id}/action")
    def game_action(session_id: str, request: ActionRequest) -> dict:
        try:
            return service.submit_action(session_id, request.action_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND, detail="Unknown or expired session"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc


def create_app(
    static_dir: Path,
    *,
    chart_service: ChartServiceProtocol | None = None,
    play_service: PlayService | None = None,
) -> FastAPI:
    """Create the FastAPI app serving the static UI plus any enabled API surfaces.

    At least one of ``chart_service`` / ``play_service`` is registered; the SPA is
    always served. API routes for a service are mounted only when it is provided.
    """
    if chart_service is None and play_service is None:
        raise ValueError("create_app requires a chart_service, a play_service, or both")

    app = FastAPI(title="Poker Solver API", version="1.0.0")
    index_html = static_dir / "index.html"

    if not index_html.exists():
        raise FileNotFoundError("UI build not found. Run `cd ui && npm install && npm run build`.")

    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    if chart_service is not None:
        _register_chart_routes(app, chart_service)
    if play_service is not None:
        _register_play_routes(app, play_service)

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(index_html)

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Not found")
        file_path = _resolve_static_path(static_dir, path)
        if file_path is not None:
            return FileResponse(file_path)
        return FileResponse(index_html)

    return app


class FastAPIChartServer:
    """Threaded uvicorn server for local chart viewing and/or play from the CLI."""

    def __init__(
        self,
        service: ChartService | None,
        base_dir: Path,
        port: int = 5173,
        *,
        play_service: PlayService | None = None,
    ) -> None:
        self.service = service
        self.play_service = play_service
        self.base_dir = base_dir
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        static_dir = self.base_dir / "ui" / "dist"
        app = create_app(
            static_dir,
            chart_service=self.service,
            play_service=self.play_service,
        )
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait briefly for bind/start failures.
        for _ in range(100):
            if self._server.started:
                return
            if not self._thread.is_alive():
                break
            time.sleep(0.02)
        raise OSError(f"Unable to start FastAPI server on port {self.port}")

    def stop(self) -> None:
        if not self._server:
            return
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
