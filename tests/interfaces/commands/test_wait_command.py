"""`wait` blocks in `run`, and stops for exactly two reasons.

The loop is tested with a stubbed tick and a fake clock: what it looks at is
`tasks`' business, already pinned there. What this pins is the shape a caller
leans on -- returns the tick that succeeded, never sleeps past the deadline,
and a timeout is an exit-1 refusal that says how far the target got.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from src.interfaces.commands import tasks, wait
from src.interfaces.errors import CommandError
from src.shared.task_history import TaskRow


def _row(cause: str, attempt: int = 1) -> TaskRow:
    return TaskRow(task_id="t-1", attempt=attempt, cause=cause, cause_source="node")


class _Clock:
    """Advances only when the loop sleeps, so `waited` is what the loop chose."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _poll(looks: list[wait.Seen], **over: Any) -> tuple[wait.WaitPayload, _Clock]:
    clock = _Clock()
    queue = iter(looks)
    knobs: dict[str, Any] = {"every": 30, "timeout": 100} | over
    payload = wait.poll(
        lambda: next(queue),
        target="t-1",
        condition="finished",
        clock=clock,
        sleep=clock.sleep,
        **knobs,
    )
    return payload, clock


class TestTheLoop:
    def test_a_condition_already_true_returns_on_the_first_look_without_sleeping(self):
        payload, clock = _poll([wait.Seen(reached=True, row=_row("completed"))])
        assert payload.reached
        assert payload.ticks == 1
        assert payload.waited_seconds == 0
        assert clock.slept == []
        assert payload.row is not None
        assert payload.row.cause == "completed"

    def test_it_keeps_looking_until_the_condition_holds(self):
        looks = [
            wait.Seen(reached=False, row=_row("preparing")),
            wait.Seen(reached=False, row=_row("running")),
            wait.Seen(reached=True, row=_row("completed")),
        ]
        payload, clock = _poll(looks)
        assert payload.ticks == 3
        assert clock.slept == [30, 30]
        assert payload.waited_seconds == 60

    def test_a_timeout_is_a_refusal_that_says_how_far_it_got(self):
        """Exit 1 is what a Monitor until-loop keys on; the line is for the
        human reading why. It also never starts a sleep the deadline would
        cut short -- the caller's own timeout would kill the process silently."""
        never = wait.Seen(reached=False, row=_row("running", attempt=2))
        with pytest.raises(CommandError) as refused:
            _poll([never] * 10)
        message = str(refused.value)
        assert "did not reach finished" in message
        assert "last seen running (attempt 2)" in message
        # every=30, timeout=100: looks at 0, 30, 60, 90; the sleep to 120 is not taken.
        assert "(4 look(s))" in message

    def test_timeout_zero_waits_forever(self):
        looks = [wait.Seen(reached=False)] * 50 + [wait.Seen(reached=True, rung=4_000)]
        payload, _ = _poll(looks, timeout=0)
        assert payload.ticks == 51
        assert payload.rung == 4_000


def _ns(**over: Any) -> argparse.Namespace:
    base = {"task": "", "until": "finished", "run": "", "rung": 0, "every": 30, "timeout": 540}
    return argparse.Namespace(**(base | over))


class TestTheFlagsNameOneTarget:
    def test_a_task_and_its_condition(self):
        assert wait.plan(_ns(task="t-1", until="log")) == ("t-1", "log")

    def test_a_run_and_its_rung(self):
        assert wait.plan(_ns(run="run-a", rung=4000)) == ("run-a", "rung")

    @pytest.mark.parametrize(
        "flags",
        [
            {},
            {"task": "t-1", "run": "run-a", "rung": 1},
            {"task": "t-1", "rung": 5},
            {"run": "run-a"},
        ],
        ids=["neither", "both", "task-with-rung", "run-without-rung"],
    )
    def test_the_rest_are_refused(self, flags):
        with pytest.raises(CommandError):
            wait.plan(_ns(**flags))


class TestATaskLookReadsThroughTasks:
    """Through `tasks`, with reconcile ON: a task the OOM killer took sits at
    `started` in the node's account forever, and only Batch says it is gone."""

    def _rows(self, monkeypatch, rows: list[TaskRow]) -> dict[str, Any]:
        asked: dict[str, Any] = {}

        def _invoke_as(model, **over):
            asked.update(over)
            return tasks.TasksPayload(rows=rows)

        # `Command` is frozen, so the module attribute is what gets replaced.
        monkeypatch.setattr(tasks, "COMMAND", SimpleNamespace(invoke_as=_invoke_as))
        return asked

    def test_finished_means_a_terminal_cause_on_the_latest_attempt(self, monkeypatch):
        asked = self._rows(monkeypatch, [_row("failed", 1), _row("running", 2)])
        seen = wait._task_tick("t-1", "finished")()
        assert not seen.reached
        assert seen.row is not None
        assert seen.row.attempt == 2
        assert asked == {"task": "t-1"}, "narrowed by task, and reconcile not skipped"

    def test_started_means_running_or_later(self, monkeypatch):
        self._rows(monkeypatch, [_row("preparing")])
        assert not wait._task_tick("t-1", "started")().reached
        self._rows(monkeypatch, [_row("running")])
        assert wait._task_tick("t-1", "started")().reached
        self._rows(monkeypatch, [_row("killed")])
        assert wait._task_tick("t-1", "started")().reached

    def test_a_task_the_record_has_not_seen_is_not_reached(self, monkeypatch):
        self._rows(monkeypatch, [])
        seen = wait._task_tick("t-1", "finished")()
        assert not seen.reached
        assert seen.describe() == "nothing recorded yet"
