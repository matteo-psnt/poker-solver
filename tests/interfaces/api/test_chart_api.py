"""Tests for FastAPI chart service app."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.interfaces.api.app import create_app


class _DummyChartService:
    def get_metadata(self) -> dict:
        return {
            "runId": "run-123",
            "positions": [{"id": 0, "label": "Button (BTN)"}],
            "situations": [{"id": "first_to_act", "label": "First to act"}],
            "defaultPosition": 0,
            "defaultSituation": "first_to_act",
        }

    def get_chart(self, position: int, situation_id: str) -> dict:
        return {
            "runId": "run-123",
            "position": position,
            "situation": situation_id,
            "positionLabel": "Button (BTN)",
            "situationLabel": "First to act",
            "bettingSequence": "",
            "ranks": "AKQJT98765432",
            "actions": [],
            "grid": [],
        }


def _build_static_dir(tmp_path: Path) -> Path:
    static_dir = tmp_path / "dist"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text("<html><body>viewer</body></html>")
    (assets_dir / "app.js").write_text("console.log('ok')")
    return static_dir


def test_chart_api_routes_and_static_mount(tmp_path: Path):
    app = create_app(_DummyChartService(), _build_static_dir(tmp_path))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    meta = client.get("/api/meta")
    assert meta.status_code == 200
    assert meta.json()["runId"] == "run-123"

    chart = client.get("/api/chart", params={"position": 1, "situation": "first_to_act"})
    assert chart.status_code == 200
    assert chart.json()["position"] == 1

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text

    # SPA fallback should serve index for unknown paths.
    page = client.get("/viewer/hand/AKs")
    assert page.status_code == 200
    assert "viewer" in page.text


def test_chart_api_validates_position_query(tmp_path: Path):
    app = create_app(_DummyChartService(), _build_static_dir(tmp_path))
    client = TestClient(app)

    response = client.get("/api/chart", params={"position": 2, "situation": "first_to_act"})
    assert response.status_code == 422
