"""FastAPI app factory and local server wrapper for chart viewing."""

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
from src.interfaces.api.models import ChartResponse, HealthResponse, MetaResponse


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


def create_app(service: ChartServiceProtocol, static_dir: Path) -> FastAPI:
    """Create FastAPI app for chart serving and static UI."""
    app = FastAPI(title="Poker Solver Chart API", version="1.0.0")
    index_html = static_dir / "index.html"

    if not index_html.exists():
        raise FileNotFoundError("UI build not found. Run `cd ui && npm install && npm run build`.")

    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
    """Threaded uvicorn server for local chart viewing from CLI."""

    def __init__(self, service: ChartService, base_dir: Path, port: int = 5173) -> None:
        self.service = service
        self.base_dir = base_dir
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        static_dir = self.base_dir / "ui" / "dist"
        app = create_app(self.service, static_dir)
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
        raise OSError(f"Unable to start FastAPI chart server on port {self.port}")

    def stop(self) -> None:
        if not self._server:
            return
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
