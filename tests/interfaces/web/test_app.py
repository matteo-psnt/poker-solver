"""The HTTP layer: routing, the memo, and how failures become status codes.

No network. `Command` is a frozen dataclass, so a per-instance stub is
impossible -- and patching `Command.invoke` on the CLASS is the better seam
anyway: it is exactly the contract this layer depends on, so the tests exercise
the web code and nothing else.
"""

from __future__ import annotations

from typing import Any

import pytest
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from fastapi.testclient import TestClient

from src.interfaces.cli.commands._base import Command
from src.interfaces.errors import CommandError
from src.interfaces.web import app as web_app


@pytest.fixture
def client() -> TestClient:
    web_app._cache.clear()
    return TestClient(web_app.create_app())


@pytest.fixture
def invoked(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record every `invoke`, and answer with a marker payload.

    Returns the call log, so a test can assert both what reached the command
    layer and how often.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
        calls.append((self.name, kwargs))
        return {"op": self.name, "seen": kwargs}

    monkeypatch.setattr(Command, "invoke", _invoke)
    return calls


def _raising(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(Command, "invoke", _invoke)


class TestItAnswersThroughTheCommand:
    def test_the_payload_is_returned_verbatim(self, client, invoked):
        response = client.get("/api/pool")
        assert response.status_code == 200
        assert response.json()["op"] == "pool-status"
        assert [name for name, _ in invoked] == ["pool-status"]

    def test_query_parameters_reach_the_command(self, client, invoked):
        client.get("/api/legs?limit=7")
        (_, kwargs) = invoked[0]
        assert kwargs["limit"] == 7

    def test_a_path_parameter_becomes_the_run_argument(self, client, invoked):
        client.get("/api/runs/run-production-025433-1095")
        assert invoked[0] == ("runinfo", {"run": "run-production-025433-1095"})

    def test_nested_run_routes_do_not_collide(self, client, invoked):
        """`/runs/{id}` and `/runs/{id}/curve` are different questions, and the
        more specific one must not be swallowed by the more general."""
        client.get("/api/runs/abc/curve")
        assert invoked[0][0] == "curve"


class TestFailuresBecomeStatusCodes:
    """A panel must be able to fail alone, and say why."""

    def test_a_refusal_is_422_with_the_reason(self, client, monkeypatch):
        _raising(monkeypatch, CommandError("'run-x' is not published"))
        response = client.get("/api/pool")
        assert response.status_code == 422
        assert "not published" in response.json()["error"]

    def test_an_expired_credential_is_503_and_names_the_fix(self, client, monkeypatch):
        """The single most likely failure in daily use."""
        _raising(monkeypatch, ClientAuthenticationError("nope"))
        response = client.get("/api/pool")
        assert response.status_code == 503
        assert "az login" in response.json()["error"]

    def test_an_unreachable_endpoint_is_503(self, client, monkeypatch):
        _raising(monkeypatch, HttpResponseError("gone"))
        assert client.get("/api/pool").status_code == 503

    def test_a_bug_still_propagates(self, client, monkeypatch):
        """Only known failures are translated. Laundering a ValueError into a
        tidy 503 would have the UI render a bug as 'unavailable'."""
        _raising(monkeypatch, ValueError("kaboom"))
        with pytest.raises(ValueError, match="kaboom"):
            client.get("/api/pool")


class TestTheMemo:
    def test_repeated_reads_share_one_cloud_sweep(self, client, invoked):
        """Several open tabs must not each trigger a 2-4s read."""
        for _ in range(5):
            assert client.get("/api/pool").status_code == 200
        assert len(invoked) == 1

    def test_different_arguments_are_different_entries(self, client, invoked):
        """`?limit=5` and `?limit=50` are different questions."""
        client.get("/api/jobs?limit=5")
        client.get("/api/jobs?limit=50")
        assert [kwargs["limit"] for _, kwargs in invoked] == [5, 50]

    def test_different_commands_do_not_share_an_entry(self, client, invoked):
        client.get("/api/pool")
        client.get("/api/runs")
        assert [name for name, _ in invoked] == ["pool-status", "runs"]


class TestServingTheConsole:
    def test_a_missing_build_is_reported_not_served_blank(self, client):
        """`console/dist` is gitignored, so this is what a fresh checkout sees.
        A blank page reads as a broken app rather than a skipped build step."""
        response = client.get("/legs")
        assert response.status_code == 503
        assert "not built" in response.json()["error"]

    def test_the_fallback_does_not_swallow_the_api(self, client, invoked):
        """The catch-all route is registered last and matches `/{path:path}`;
        an ordering mistake would turn every endpoint into the SPA shell."""
        assert client.get("/api/pool").status_code == 200
        assert invoked
