"""The composed views: what they ask for, and what they join.

No network, same seam as `test_app.py` -- `Command.invoke` is patched on the
class, so these exercise the composition and nothing underneath it.

The properties worth pinning are the ones a passing screen would not reveal.
A view that quietly dropped a part still renders; a view whose join silently
returns `[]` because a part FAILED renders too, and says "this run has no
tasks", which is a different and worse thing than saying nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.interfaces.web import app as web_app
from src.interfaces.web import views

TASK_ROWS = [
    {"task_id": "t1", "run_id": "run-a", "cause": "ok"},
    {"task_id": "t2", "run_id": "run-b", "cause": "died"},
    {"task_id": "t3", "run_id": "run-a", "cause": "ok"},
    {"task_id": "t4", "run_id": None, "cause": "ok"},
]

RUN_ROWS = [
    {"name": "run-a", "experiment_id": "exp-1", "arm": "control"},
    {"name": "run-b", "experiment_id": "exp-1", "arm": "variant"},
    {"name": "run-c", "experiment_id": "exp-2", "arm": "control"},
    {"name": "run-d", "experiment_id": None, "arm": None},
]


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record every invoke and answer with a plausible payload for its command."""
    calls: list[tuple[str, dict[str, Any]]] = []
    bodies: dict[str, dict[str, Any]] = {
        "tasks": {"op": "tasks", "rows": TASK_ROWS},
        "runs": {"op": "runs", "runs": RUN_ROWS},
    }

    def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
        calls.append((self.name, kwargs))
        return bodies.get(self.name, {"op": self.name, "seen": kwargs})

    monkeypatch.setattr(Command, "invoke", _invoke)
    return calls


def _failing(monkeypatch: pytest.MonkeyPatch, *, only: str) -> None:
    """Make ONE command unavailable, leaving the others answering."""

    def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
        if self.name == only:
            raise CommandError(f"{only} is unavailable")
        return {"op": self.name, "rows": TASK_ROWS, "runs": RUN_ROWS}

    monkeypatch.setattr(Command, "invoke", _invoke)


class TestOneScreenIsOneRequest:
    def test_now_asks_every_panel_it_shows(self, answers):
        composed = views.now()
        assert sorted(name for name, _ in answers) == [
            "autoscale-check",
            "cost",
            "jobs",
            "pool-status",
            "tasks",
        ]
        assert set(composed["parts"]) == {"pool", "jobs", "tasks", "autoscale", "cost"}

    def test_a_run_page_is_five_questions_in_one(self, answers):
        composed = views.run("run-a")
        assert sorted(name for name, _ in answers) == [
            "curve",
            "ledger",
            "progress",
            "runinfo",
            "tasks",
        ]
        assert set(composed["parts"]) == {"run", "progress", "curve", "evals", "tasks"}

    def test_the_run_id_reaches_every_part_that_takes_one(self, answers):
        views.run("run-a")
        asked = dict(answers)
        assert asked["runinfo"]["run"] == "run-a"
        assert asked["progress"]["run"] == "run-a"
        assert asked["curve"]["run"] == "run-a"
        # `ledger` filters by run ITSELF -- it has the flag, so the view must use
        # it rather than fetch everything and filter here.
        assert asked["ledger"]["run"] == "run-a"

    def test_the_full_checkpoint_history_is_requested(self, answers):
        """`runinfo` truncates progress to its `--last` default of eight, so the
        console's chart drew 8 of 112 checkpoints and looked complete."""
        views.run("run-a")
        asked = dict(answers)
        assert asked["progress"]["last"] == 0


class TestTheJoins:
    def test_a_run_s_tasks_are_drawn_out_of_the_full_log(self, answers):
        """`tasks` has no `--run` flag, so this join is the view's own work --
        and it is the one the browser used to do after downloading everything."""
        composed = views.run("run-a")
        assert [row["task_id"] for row in composed["run_tasks"]] == ["t1", "t3"]

    def test_a_task_belonging_to_no_run_is_not_swept_in(self, answers):
        composed = views.run("run-a")
        assert all(row["run_id"] == "run-a" for row in composed["run_tasks"])

    def test_the_full_task_log_does_not_go_on_the_wire(self, answers):
        """The whole point of joining here. Filtering server-side while still
        shipping every row under `parts` would move the work and keep the bytes
        -- and the bytes are what made this page slow."""
        composed = views.run("run-a")
        assert composed["parts"]["tasks"]["payload"]["rows"] == []
        assert composed["parts"]["tasks"]["payload"]["source_rows"] == len(TASK_ROWS)
        assert len(composed["run_tasks"]) == 2

    def test_trimming_a_part_does_not_edit_what_the_command_returned(self, answers):
        """The payload is memoised per (command, arguments) and shared by every
        reader for the TTL. A view that empties `rows` in place hands the next
        caller of `/api/tasks` an empty task log -- and hands ITSELF one on the
        second read, so the page is correct exactly once.

        Asserted as two successive views of DIFFERENT runs, because that is how
        it presents: run-a renders, then run-b is blank, and nothing about the
        second request looks wrong.
        """
        first = views.run("run-a")
        second = views.run("run-b")
        assert [row["task_id"] for row in first["run_tasks"]] == ["t1", "t3"]
        assert [row["task_id"] for row in second["run_tasks"]] == ["t2"]
        assert TASK_ROWS[0]["task_id"] == "t1", "the module-level fixture was mutated"

    def test_the_discarded_part_still_carries_its_failure(self, monkeypatch):
        """Trimming the rows must not trim the reason: `_summarise_rows` runs
        over a part that may have no payload at all."""
        _failing(monkeypatch, only="tasks")
        composed = views.run("run-a")
        assert composed["parts"]["tasks"]["error"] == "tasks is unavailable"

    def test_the_arms_of_one_experiment_are_pinned_to_their_run_records(self, answers):
        composed = views.experiment("exp-1")
        assert [row["name"] for row in composed["arm_runs"]] == ["run-a", "run-b"]

    def test_an_untagged_run_is_not_an_arm_of_everything(self, answers):
        """`experiment_id: None` must not match an experiment id."""
        composed = views.experiment("exp-2")
        assert [row["name"] for row in composed["arm_runs"]] == ["run-c"]


class TestOnePartCannotTakeOutTheScreen:
    def test_the_other_panels_survive_an_unavailable_one(self, monkeypatch):
        _failing(monkeypatch, only="pool-status")
        composed = views.now()
        assert composed["parts"]["pool"]["error"] == "pool-status is unavailable"
        assert composed["parts"]["jobs"]["payload"] is not None
        assert composed["parts"]["cost"]["payload"] is not None

    def test_a_failed_part_is_reported_not_joined_away(self, monkeypatch):
        """The trap this test exists for: the join returns `[]` when `tasks` is
        unavailable, and `[]` renders as "no tasks for this run" -- a confident
        wrong answer. The part's error is what the UI must show instead, so it
        has to still be there.
        """
        _failing(monkeypatch, only="tasks")
        composed = views.run("run-a")
        assert composed["run_tasks"] == []
        assert composed["parts"]["tasks"]["error"] == "tasks is unavailable"
        assert composed["parts"]["run"]["payload"] is not None


class TestOverHttp:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(web_app.create_app())

    def test_the_now_view_is_served(self, client, answers):
        response = client.get("/api/view/now")
        assert response.status_code == 200
        assert response.json()["op"] == "view-now"

    def test_the_run_view_carries_its_join(self, client, answers):
        response = client.get("/api/view/run/run-a")
        assert response.status_code == 200
        assert [row["task_id"] for row in response.json()["run_tasks"]] == ["t1", "t3"]

    def test_two_runs_do_not_share_a_cache_entry(self, client, answers):
        """The cache key is (view, arguments). A view keyed on its name alone
        would serve run-a's page for run-b, which is the worst kind of wrong:
        plausible, and about the thing you are trying to read."""
        first = client.get("/api/view/run/run-a").json()
        second = client.get("/api/view/run/run-b").json()
        assert [row["task_id"] for row in first["run_tasks"]] == ["t1", "t3"]
        assert [row["task_id"] for row in second["run_tasks"]] == ["t2"]

    def test_a_repeat_request_is_served_from_the_memo(self, client, answers):
        client.get("/api/view/now")
        before = len(answers)
        client.get("/api/view/now")
        assert len(answers) == before, "the second request re-ran the whole fan-out"

    def test_an_azure_outage_is_a_503_not_a_500(self, client, monkeypatch):
        """A view fails as a whole only when the fan-out itself cannot run --
        `attempt` still classifies it, so the client can tell 'retry' from
        'you asked for something that does not exist'."""
        from azure.core.exceptions import ClientAuthenticationError

        def _invoke(self: Command, **kwargs: Any) -> dict[str, Any]:
            raise ClientAuthenticationError("expired")

        monkeypatch.setattr(Command, "invoke", _invoke)
        # Every part fails independently, so the VIEW still answers 200 with
        # five greyed panels. That is the design: a screen that survives.
        response = client.get("/api/view/now")
        assert response.status_code == 200
        assert all(part["error"] for part in response.json()["parts"].values())
