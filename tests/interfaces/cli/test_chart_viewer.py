"""Tests for chart viewer FastAPI server flow and control."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.interfaces.cli.flows.chart import viewer
from src.interfaces.cli.ui.context import CliContext


def _make_ctx(tmp_path):
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def _fake_chart_service(num_infosets: int = 10):
    runtime = SimpleNamespace(storage=SimpleNamespace(num_infosets=lambda: num_infosets))
    return SimpleNamespace(runtime=runtime)


def test_viewer_uses_fastapi_server_by_default(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    calls = {"fastapi": 0}

    class _FastAPIServer:
        base_url = "http://127.0.0.1:5173"

        def __init__(self, service, base_dir):
            assert service.runtime.storage.num_infosets() == 10
            calls["fastapi"] += 1

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(viewer.services, "list_runs", lambda _runs_dir: ["run-a"])
    monkeypatch.setattr(viewer.prompts, "select", lambda *_args, **_kwargs: "run-a")
    monkeypatch.setattr(viewer, "_ensure_ui_build", lambda _ctx: True)
    monkeypatch.setattr(
        viewer.ChartService, "from_run_dir", lambda *_args, **_kwargs: _fake_chart_service()
    )
    monkeypatch.setattr(viewer, "FastAPIChartServer", _FastAPIServer)
    monkeypatch.setattr(viewer.ui, "pause", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(viewer.ui, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(viewer.webbrowser, "open", lambda *_args, **_kwargs: True)

    viewer.view_preflop_chart(ctx)

    assert calls["fastapi"] == 1
