"""The HTTP layer: routing, the memo, and how failures become status codes.

No network. `Command` is a frozen dataclass, so a per-instance stub is
impossible -- and patching `Command.invoke` on the CLASS is the better seam
anyway: it is exactly the contract this layer depends on, so the tests exercise
the web code and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from fastapi.testclient import TestClient

from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.interfaces.web import app as web_app


@pytest.fixture
def client() -> TestClient:
    """A fresh application, and therefore a fresh memo.

    Nothing to reset: the cache belongs to the app `create_app` built, so it
    cannot outlive this fixture or be seen by the next test.
    """
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

    def test_the_memo_belongs_to_the_application(self, client, invoked):
        """A second app must not be born holding the first one's answers.

        Module-level state would make this pass by accident and make every test
        above depend on the order it ran in.
        """
        client.get("/api/pool")
        assert TestClient(web_app.create_app()).get("/api/pool").status_code == 200
        assert len(invoked) == 2


class TestServingTheConsole:
    """Both branches, driven explicitly.

    These used to assert whichever state the working tree happened to be in,
    which meant the "not built" test passed until someone built the console and
    then failed for a reason that had nothing to do with the code. `dist/` is
    gitignored, so its presence is never a property a test can rely on.
    """

    def _app_with_dist(self, monkeypatch, dist: Path) -> TestClient:
        monkeypatch.setattr(web_app, "CONSOLE_DIST", dist)
        return TestClient(web_app.create_app())

    def test_a_missing_build_is_reported_not_served_blank(self, tmp_path, monkeypatch):
        """A blank page reads as a broken app rather than a skipped build step."""
        client = self._app_with_dist(monkeypatch, tmp_path / "absent")
        response = client.get("/legs")
        assert response.status_code == 503
        assert "not built" in response.json()["error"]

    def test_a_client_routed_path_gets_the_shell(self, tmp_path, monkeypatch):
        """`/legs` and `/runs/abc` are routes, not files. Returning 404 for them
        is the classic SPA deployment bug: the app works until it is reloaded."""
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html><div id=root></div>")
        client = self._app_with_dist(monkeypatch, dist)

        for path in ("/", "/legs", "/runs/run-production-025433-1095"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert "id=root" in response.text

    def test_the_fallback_does_not_swallow_the_api(self, client, invoked):
        """The catch-all route is registered last and matches `/{path:path}`;
        an ordering mistake would turn every endpoint into the SPA shell."""
        assert client.get("/api/pool").status_code == 200
        assert invoked
