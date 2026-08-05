"""The console's blueprint endpoints forward, and fail in distinguishable ways.

"Not configured" and "configured but unreachable" are different problems with
different fixes, and a bare connection error conflates them -- so both are
asserted here rather than left to whichever one happens to be true on a laptop.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.interfaces.web import blueprint_proxy


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv(blueprint_proxy.BLUEPRINT_URL_ENV, raising=False)
    app = FastAPI()
    blueprint_proxy.mount(app)
    return TestClient(app)


class TestWhenNothingIsConfigured:
    @pytest.mark.parametrize("route", ["/api/blueprint/run", "/api/blueprint/node"])
    def test_it_says_so_rather_than_failing_to_connect(self, client, route):
        response = client.get(route)

        assert response.status_code == 503
        assert blueprint_proxy.BLUEPRINT_URL_ENV in response.json()["error"]


class TestWhenTheServerIsUnreachable:
    def test_the_address_is_named_in_the_error(self, client, monkeypatch):
        monkeypatch.setenv(blueprint_proxy.BLUEPRINT_URL_ENV, "http://127.0.0.1:9")

        def _refuse(*_args, **_kwargs):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", _refuse)
        response = client.get("/api/blueprint/run")

        assert response.status_code == 503
        assert "127.0.0.1:9" in response.json()["error"]


class TestForwarding:
    def test_a_refusal_is_passed_through_unchanged(self, client, monkeypatch):
        """A 422 was already phrased for a person by the analysis layer; rewriting
        it here would mean maintaining a second vocabulary for one failure."""
        monkeypatch.setenv(blueprint_proxy.BLUEPRINT_URL_ENV, "http://server")
        monkeypatch.setattr(
            httpx,
            "get",
            lambda *_a, **_k: httpx.Response(422, json={"error": "'b9' is not available here."}),
        )
        response = client.get("/api/blueprint/node", params={"path": "b9"})

        assert response.status_code == 422
        assert response.json()["error"] == "'b9' is not available here."

    def test_the_query_is_forwarded(self, client, monkeypatch):
        monkeypatch.setenv(blueprint_proxy.BLUEPRINT_URL_ENV, "http://server/")
        seen: dict[str, object] = {}

        def _capture(url, params=None, timeout=None):
            seen["url"] = url
            seen["params"] = params
            return httpx.Response(200, json={"ok": True})

        monkeypatch.setattr(httpx, "get", _capture)
        client.get("/api/blueprint/node", params={"path": "c/x", "board": "2c7d9h"})

        # The trailing slash on the base must not survive into the path.
        assert seen["url"] == "http://server/api/node"
        assert seen["params"] == {"path": "c/x", "board": "2c7d9h", "average": True}

    def test_a_server_fault_becomes_a_503_not_a_500(self, client, monkeypatch):
        """The console is up; the thing it asked is not. That is a 503 to a client."""
        monkeypatch.setenv(blueprint_proxy.BLUEPRINT_URL_ENV, "http://server")
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: httpx.Response(500, text="boom"))
        response = client.get("/api/blueprint/run")

        assert response.status_code == 503
