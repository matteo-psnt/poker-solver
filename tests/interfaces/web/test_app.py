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
        client.get("/api/tasks?limit=7")
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


class TestTheDispatchingWrites:
    """The seven writes that queue work or move the record.

    Every one is still a single `Command.invoke` -- pinned structurally by
    `test_no_second_read_path` -- so what is left to check is the body contract:
    that a field the caller omitted does not arrive as a value the command line
    would never have produced.
    """

    def test_a_body_reaches_the_command_as_arguments(self, client, invoked):
        client.post("/api/submit", json={"to": 25_000_000, "config": "production"})
        assert invoked[0] == ("submit", {"to": 25_000_000, "config": "production"})

    def test_an_omitted_field_is_not_sent_at_all(self, client, invoked):
        """The property the whole body design rests on.

        Sending `config: ""` would be harmless; sending `workers: 0` instead of
        omitting it would pin a 32-core node to one worker. Neither is decided
        here -- the command's own parser holds every default, and it only gets
        to apply them if the key is absent.
        """
        client.post("/api/submit", json={"to": 1})
        (_, kwargs) = invoked[0]
        assert kwargs == {"to": 1}

    def test_a_falsy_value_that_was_given_is_still_sent(self, client, invoked):
        """Omitted and `false` are different answers, and `given` drops only the
        first. Collapsing them would make `--force` unreachable in reverse: a
        caller could never explicitly say no."""
        client.post("/api/precompute", json={"config": "production", "force": False})
        (_, kwargs) = invoked[0]
        assert kwargs == {"config": "production", "force": False}

    def test_a_missing_required_field_is_refused_before_the_command(self, client, invoked):
        """422 from the model, and nothing dispatched."""
        assert client.post("/api/promote", json={"run": "run-a"}).status_code == 422
        assert not invoked

    def test_compacting_defaults_to_the_dry_run(self, client, invoked):
        """`--delete` is the irreversible half. An empty body must not reach it.

        Not by writing `False` here -- by sending nothing, so argparse's
        `store_true` default is what answers.
        """
        client.post("/api/compact-legs", json={})
        (_, kwargs) = invoked[0]
        assert "delete" not in kwargs
        assert "apply" not in kwargs

    def test_a_dispatch_is_never_answered_from_the_memo(self, client, invoked):
        """Two identical submissions are two runs someone wants.

        The read memo keys on (command, arguments), so an unqualified `answer`
        would report the first job's id for a task that was never queued.
        """
        for _ in range(3):
            client.post("/api/submit", json={"to": 1})
        assert len(invoked) == 3

    def test_a_write_is_unreachable_by_get(self, client, invoked):
        """A GET that queues a cloud task is reachable from a link preview.

        The status is 404 rather than 405 because the SPA fallback matches every
        GET: an `/api` path that got that far named no endpoint, which is what
        it should say. What matters is the second assertion — nothing dispatched.
        """
        for path in (
            "/api/submit",
            "/api/score",
            "/api/precompute",
            "/api/push-code",
            "/api/push-data",
            "/api/compact-legs",
            "/api/promote",
        ):
            assert client.get(path).status_code == 404, path
        assert not invoked


class TestTheReadsAddedForCoverage:
    def test_a_flag_named_like_answers_own_parameter_still_reaches_the_command(
        self, client, invoked
    ):
        """`activity --command tasks` is the case, and it is not a one-off.

        `answer(cache, command, /, **kwargs)` — the slash is what keeps a
        command's own flags from binding to this function's parameters. Without
        it the failure is a type error several frames from anything the reader
        was thinking about, and the same trap waits for a future `--cache`.
        """
        client.get("/api/activity?days=1")
        (name, kwargs) = invoked[0]
        assert name == "activity"
        assert kwargs["command"] == ""
        assert kwargs["days"] == 1

    def test_an_experiment_id_becomes_the_report_argument(self, client, invoked):
        client.get("/api/experiments/exp-7")
        assert invoked[0] == ("report", {"experiment": "exp-7"})

    def test_an_unspecified_rung_reaches_compare_as_none(self, client, invoked):
        """`--a-at`'s own default. `0` would be read as a rung and match nothing."""
        client.get("/api/compare?a=run-a&b=run-b")
        (_, kwargs) = invoked[0]
        assert kwargs == {"a": "run-a", "b": "run-b", "a_at": None, "b_at": None, "force": False}


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
        response = client.get("/tasks")
        assert response.status_code == 503
        assert "not built" in response.json()["error"]

    def test_a_client_routed_path_gets_the_shell(self, tmp_path, monkeypatch):
        """`/tasks` and `/runs/abc` are routes, not files. Returning 404 for them
        is the classic SPA deployment bug: the app works until it is reloaded."""
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html><div id=root></div>")
        client = self._app_with_dist(monkeypatch, dist)

        for path in ("/", "/tasks", "/runs/run-production-025433-1095"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert "id=root" in response.text

    def test_an_unmatched_api_path_is_a_404_not_the_shell(self, tmp_path, monkeypatch):
        """Both branches of the fallback, because both can swallow one.

        Serving the console for `/api/typo` reports 200 and HTML for a request
        that found nothing, which reads as a broken endpoint rather than a
        wrong URL — and is what a `curl` against a write endpoint would see.
        """
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><div id=root></div>")

        for client in (
            self._app_with_dist(monkeypatch, dist),
            self._app_with_dist(monkeypatch, tmp_path / "absent"),
        ):
            response = client.get("/api/no-such-thing")
            assert response.status_code == 404
            assert "No such endpoint" in response.json()["error"]

    def test_the_fallback_does_not_swallow_the_api(self, client, invoked):
        """The catch-all route is registered last and matches `/{path:path}`;
        an ordering mistake would turn every endpoint into the SPA shell."""
        assert client.get("/api/pool").status_code == 200
        assert invoked


class TestThePayloadEncodesLikeTheCLI:
    """The bytes are the interface, not just the dict.

    `headless --json` serialises with `jsonio.dumps`, which carries a `default`
    hook for values JSON has no native form for. Starlette's `JSONResponse` has
    no such hook, so the same payload printed fine on one surface and raised a
    `TypeError` from inside the response on the other -- past the refusal /
    unavailable ladder, reaching the browser as an untranslated 500.

    Nothing reaches it today: every console-reachable payload is derived from
    JSONL on the share, and the numpy-adjacent numbers are coerced at their
    source. That is provenance, not a guarantee, and `PAYLOADS` cannot pin it --
    the fixture is hand-written JSON-native literals, so a probe over it passes
    unconditionally. Hence a payload built to be hostile.
    """

    def test_a_payload_json_cannot_natively_encode_still_answers(self, client, monkeypatch):
        import numpy as np

        def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
            return {
                "op": "pool-status",
                "coverage": np.float32(0.5),
                "touched": np.int64(7),
                "where": Path("/mnt/work/runs"),
            }

        monkeypatch.setattr(Command, "invoke", _invoke)
        response = client.get("/api/pool")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["coverage"] == pytest.approx(0.5)
        assert body["touched"] == pytest.approx(7)
        assert body["where"] == "/mnt/work/runs"

    def test_a_nan_fails_loudly_rather_than_reaching_the_client(self, client, monkeypatch):
        """The one respect in which the two surfaces still differ, deliberately.

        `NaN` is not valid JSON and `JSON.parse` rejects it, so the console asks
        for a failure with an origin rather than a parse error in the browser.
        This is Starlette's own `allow_nan=False`, kept rather than introduced.

        `allow_nan=False` rejects `Infinity` too, so blessing it means knowing
        no console-reachable payload can produce one. Checked: the only ratio on
        that surface is `ExploitabilityCurve.decay_ratio`, which returns `None`
        rather than dividing when the last point is 0. What remains is a
        non-finite value already written INTO a record, which is a bug upstream
        — and failing at the boundary is how it gets an origin.
        """

        def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
            return {"op": "pool-status", "rate": float("nan")}

        monkeypatch.setattr(Command, "invoke", _invoke)
        with pytest.raises(ValueError, match="Out of range float"):
            client.get("/api/pool")
