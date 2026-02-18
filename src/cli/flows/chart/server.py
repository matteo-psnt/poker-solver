"""Local chart server for the preflop viewer UI."""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.api.chart_service import ChartService


class ChartServer:
    def __init__(
        self,
        chart_service: ChartService,
        base_dir: Path,
        port: int = 5173,
    ) -> None:
        self.chart_service = chart_service
        self.base_dir = base_dir
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        static_dir = self.base_dir / "ui" / "dist"
        if not static_dir.exists():
            raise FileNotFoundError(
                "UI build not found. Run `cd ui && npm install && npm run build`."
            )
        handler = _make_handler(self, static_dir)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._httpd:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2)
        self._httpd = None
        self._thread = None

    def get_metadata(self) -> dict:
        return self.chart_service.get_metadata()

    def get_chart(self, position: int, situation_id: str) -> dict:
        return self.chart_service.get_chart(position, situation_id)


def _make_handler(server: ChartServer, static_dir: Path):
    class ChartRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api(parsed)
                return

            self._handle_static(parsed.path)

        def _handle_api(self, parsed):
            if parsed.path == "/api/meta":
                self._send_json(server.get_metadata())
                return

            if parsed.path == "/api/chart":
                params = parse_qs(parsed.query)
                position = int(params.get("position", ["0"])[0])
                situation = params.get("situation", ["first_to_act"])[0]
                self._send_json(server.get_chart(position, situation))
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

        def _handle_static(self, path: str) -> None:
            if path == "/":
                file_path = static_dir / "index.html"
            else:
                safe_path = Path(unquote(path.lstrip("/")))
                file_path = (static_dir / safe_path).resolve()

                try:
                    file_path.relative_to(static_dir.resolve())
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN, "Forbidden path")
                    return

            if not file_path.exists() or file_path.is_dir():
                file_path = static_dir / "index.html"

            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "UI build not found")
                return

            content_type, _ = mimetypes.guess_type(file_path.as_posix())
            if content_type is None:
                content_type = "application/octet-stream"

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with file_path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _send_json(self, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            return

    return ChartRequestHandler
