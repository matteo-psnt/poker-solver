"""The composed status screen.

It is the first consumer of the command seam, and it exists to answer one
question -- "what is the pool doing right now" -- that previously required
running three commands and joining them by hand.

The property under test throughout is ISOLATION. A status screen whose value is
that it still tells you two things when the third is unavailable is worth
nothing if one expired credential blanks it, and that failure only appears when
something is already wrong -- exactly when nobody is in a position to debug the
dashboard.
"""

from __future__ import annotations

import argparse
from typing import Any

from azure.core.exceptions import ClientAuthenticationError

from src.interfaces.cli import headless
from src.interfaces.commands import status, tasks
from src.interfaces.commands._base import Command
from src.shared.task_history import TaskRow


def _command(name: str, run: Any) -> Command:
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--limit", type=int, default=10)

    return Command(name=name, add_arguments=add_arguments, run=run, render=lambda _p: None)


def _ok(name: str) -> Command:
    return _command(name, lambda args: {"op": name, "limit": args.limit})


def _raising(name: str, error: BaseException) -> Command:
    def _run(_args):
        raise error

    return _command(name, _run)


class TestOnePanelCannotTakeOutTheOthers:
    """What ISOLATION means for this screen specifically.

    The per-part half of it -- which exceptions become an unavailable panel and
    which still propagate -- moved to `test_compose.py` with the fan-out it
    tests. What stays here is the property that belongs to `status` rather than
    to the machinery: that THESE three panels are gathered such that one of them
    failing leaves the other two answered.
    """

    def test_the_other_panels_still_answer(self, monkeypatch):
        monkeypatch.setattr(
            status,
            "PANELS",
            (
                ("pool", _ok("pool")),
                ("jobs", _raising("jobs", ClientAuthenticationError("nope"))),
                ("tasks", _ok("tasks")),
            ),
        )
        payload = status.gather()

        assert set(payload["panels"]) == {"pool", "jobs", "tasks"}
        assert payload["panels"]["jobs"]["payload"] is None
        assert payload["panels"]["pool"]["payload"]["op"] == "pool"
        assert payload["panels"]["tasks"]["payload"]["op"] == "tasks"


class TestGatherComposesRatherThanReads:
    def test_panel_arguments_reach_the_commands(self, monkeypatch):
        monkeypatch.setattr(status, "PANELS", (("jobs", _ok("jobs")), ("tasks", _ok("tasks"))))
        payload = status.gather(limit=3)
        assert payload["panels"]["jobs"]["payload"]["limit"] == 3
        # `tasks` is limited too: it defaults to the whole history on purpose, and
        # a glanceable screen cannot carry it.
        assert payload["panels"]["tasks"]["payload"]["limit"] == 3

    def test_tasks_can_be_skipped(self, monkeypatch):
        """It is the slowest panel by a wide margin (measured: 23s vs 0.9s)."""
        monkeypatch.setattr(status, "PANELS", (("pool", _ok("pool")), ("tasks", _ok("tasks"))))
        assert set(status.gather(with_tasks=False)["panels"]) == {"pool"}


class TestWatch:
    def _args(self, watch: int) -> argparse.Namespace:
        return argparse.Namespace(watch=watch, limit=5, no_tasks=True)

    def test_an_interval_below_a_full_cycle_is_raised(self, monkeypatch):
        """A tick that cannot finish before the next is due is not a refresh
        interval, it is a queue -- every panel would show a different instant."""
        monkeypatch.setattr(status, "PANELS", (("pool", _ok("pool")),))
        assert status.run(self._args(5))["watch"] == status.MIN_INTERVAL

    def test_zero_means_print_once(self, monkeypatch):
        monkeypatch.setattr(status, "PANELS", (("pool", _ok("pool")),))
        assert status.run(self._args(0))["watch"] == 0

    def test_render_returns_without_watching_when_not_asked(self, monkeypatch, capsys):
        """`render` carries the loop, so 'does it terminate' is a real question."""
        monkeypatch.setattr(status, "PANELS", (("pool", _ok("pool")),))
        payload = status.run(self._args(0))
        monkeypatch.setitem(status.PANEL_RENDERERS, "pool", lambda p: print(p["op"]))
        status.render(payload)
        assert "pool" in capsys.readouterr().out

    def test_ctrl_c_leaves_the_loop_without_a_traceback(self, monkeypatch):
        """Ctrl-C is the documented way out, so it is a normal exit.

        Uncaught it unwinds through `headless.main`, which translates only
        `CommandError` -- so stopping a watch printed a traceback every time.
        A `timeout`-based probe never catches this: SIGTERM is not SIGINT.
        """
        monkeypatch.setattr(status, "PANELS", (("pool", _ok("pool")),))
        monkeypatch.setitem(status.PANEL_RENDERERS, "pool", lambda _p: None)

        def _interrupt(_seconds):
            raise KeyboardInterrupt

        monkeypatch.setattr(status.time, "sleep", _interrupt)
        assert headless.main(["status", "--watch", "30", "--no-tasks"]) == 0


class TestRenderDelegates:
    """No formatting of its own: a second renderer for the same payload could
    disagree with the command that owns it."""

    def test_a_panel_is_rendered_by_its_owning_command(self, monkeypatch, capsys):
        monkeypatch.setitem(status.PANEL_RENDERERS, "pool", lambda _p: print("OWNED"))
        status.render(
            {
                "at": "now",
                "elapsed_seconds": 1.0,
                "watch": 0,
                "panels": {"pool": {"payload": {}, "error": None}},
            }
        )
        assert "OWNED" in capsys.readouterr().out

    def test_a_failed_panel_says_so_instead(self, capsys):
        status.render(
            {
                "at": "now",
                "elapsed_seconds": 1.0,
                "watch": 0,
                "panels": {"pool": {"payload": None, "error": "boom"}},
            }
        )
        assert "unavailable: boom" in capsys.readouterr().out


class TestTasksLimit:
    """Added for the status screen, but `tasks` keeps its own default of ALL.

    Truncating a death log by default would hide the row being looked for.
    """

    @staticmethod
    def _rows(count: int) -> list[TaskRow]:
        """Attempts distinguishable by `attempt`, which is what the trim orders on."""
        return [
            TaskRow(task_id="t", attempt=i, cause="completed", cause_source="node")
            for i in range(count)
        ]

    def test_the_default_hides_nothing(self):
        rows = self._rows(30)
        assert tasks._result(rows, None, 0).rows == rows

    def test_a_limit_keeps_the_most_recent_and_counts_the_rest(self):
        result = tasks._result(self._rows(30), None, 4)
        assert [row.attempt for row in result.rows] == [26, 27, 28, 29]
        assert result.hidden_rows == 26
