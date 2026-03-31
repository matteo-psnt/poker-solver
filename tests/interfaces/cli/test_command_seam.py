"""The seam a second surface sits on.

Two properties make a new interface cheap, and neither held before this. A
command must be answerable WITHOUT a command line -- ``run`` takes an
``argparse.Namespace``, so the only way to ask was to parse ``sys.argv`` or to
hand-build a Namespace and hope it matched the parser. And a refusal must be a
VALUE -- 16 sites raised ``SystemExit``, which ends the process, so a long-lived
view polling several commands would be killed by the first run with no
checkpoint history.

The guard at the bottom is the one that keeps this true: a new command written
in the old idiom would pass every other test in the suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from src.interfaces.cli import headless
from src.interfaces.cli.commands import COMMANDS, progress
from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud.config import CloudConfigError
from src.interfaces.errors import CommandError


def _add(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--all", action="store_true")


def _echo(args: argparse.Namespace) -> dict[str, Any]:
    return {"op": "echo", "name": args.name, "limit": args.limit, "all": args.all}


ECHO = Command(name="echo", add_arguments=_add, run=_echo, render=lambda _p: None, help="")


class TestInvokeBuildsArgumentsFromTheParser:
    """One declaration of what a command accepts, reused rather than restated.

    A surface that re-declared the flags could drift from the parser, and the
    drift would surface as a missing key at render time -- the exact failure the
    ``Command`` dataclass was introduced to make impossible.
    """

    def test_defaults_come_from_the_parser(self):
        assert ECHO.invoke(name="x") == {"op": "echo", "name": "x", "limit": 25, "all": False}

    def test_overrides_win(self):
        payload = ECHO.invoke(name="x", limit=3, all=True)
        assert (payload["limit"], payload["all"]) == (3, True)

    def test_an_unknown_argument_is_refused_not_ignored(self):
        """Silently dropping it would read as a command ignoring its own flag."""
        with pytest.raises(CommandError, match="no such argument"):
            ECHO.invoke(name="x", limmit=3)

    def test_a_missing_required_argument_is_refused_here(self):
        """Rather than arriving as a `None` several frames into `run`."""
        with pytest.raises(CommandError, match="missing required"):
            ECHO.invoke(limit=3)

    def test_every_registered_command_can_be_introspected(self):
        """`invoke` is only usable if this holds for all of them, not just one."""
        for command in COMMANDS:
            assert isinstance(command.arguments.__self__, Command)
            with pytest.raises(CommandError, match="no such argument"):
                command.invoke(definitely_not_a_flag=1)


class TestARefusalIsAValue:
    def test_a_real_command_refuses_without_exiting(self, tmp_path):
        """End-to-end: the wiring and the error channel in one call."""
        with pytest.raises(CommandError, match="Run not found"):
            progress.COMMAND.invoke(run="nope", runs_dir=str(tmp_path), source="local", last=25)

    def test_a_cloud_config_failure_is_one_too(self):
        """So a surface catches ONE type, not a list that grows per module."""
        assert issubclass(CloudConfigError, CommandError)


class TestTheCommandLinePutsTheExitBack:
    """The CLI keeps its old behaviour; it just no longer imposes it on others."""

    def test_a_refusal_is_exit_1_and_a_message_on_stderr(self, tmp_path, capsys):
        code = headless.main(["progress", "--run", "nope", "--runs-dir", str(tmp_path)])
        assert code == 1
        assert "Run not found" in capsys.readouterr().err

    def test_a_bug_still_tracebacks(self, monkeypatch, tmp_path):
        """Only CommandError is translated. A ValueError is a bug, and a
        traceback is the correct output for one."""

        def _boom(*_args, **_kwargs):
            raise ValueError("kaboom")

        monkeypatch.setattr(progress, "resolve_run_dir", _boom)
        with pytest.raises(ValueError, match="kaboom"):
            headless.main(["progress", "--run", "r", "--runs-dir", str(tmp_path)])


def test_no_command_signals_a_refusal_by_exiting_the_process():
    """The regression guard.

    `raise SystemExit` is the idiom every one of these modules used, so a new
    command copied from an old one reintroduces it silently -- it would still
    pass on the command line, and only break the surfaces that are not one.
    """
    roots = [Path("src/interfaces/cli/commands"), Path("src/interfaces/cloud")]
    offenders = [
        str(path)
        for root in roots
        for path in root.glob("*.py")
        if "raise SystemExit" in path.read_text()
    ]
    assert offenders == [], f"raise CommandError instead: {offenders}"
