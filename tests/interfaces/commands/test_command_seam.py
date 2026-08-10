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
from src.interfaces.cloud.config import CloudConfigError
from src.interfaces.commands import load_all, progress
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError


def _add(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--all", action="store_true")


def _echo(args: argparse.Namespace) -> dict[str, Any]:
    return {"op": "echo", "name": args.name, "limit": args.limit, "all": args.all}


ECHO = Command(name="echo", add_arguments=_add, run=_echo, render=lambda _p: None, help="")


def _placeholder(action: argparse.Action) -> Any:
    """A value a required flag will actually accept.

    Type-aware because the comparison below has to survive the parser's own
    coercion: handing `"x"` to `--to`, which is `type=int`, makes argparse exit
    2 and proves nothing about the seam.
    """
    if action.choices:
        return next(iter(action.choices))
    return action.type("1") if callable(action.type) else "x"


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

    @pytest.mark.parametrize("command", load_all(), ids=lambda c: c.name)
    def test_invoke_and_the_command_line_agree_for_every_command(self, command: Command):
        """The contract, stated exactly: the two paths must build the SAME args.

        Reading the parser's declared defaults is not the same as letting
        argparse parse, and where they disagree the command line is right and
        the other surfaces are silently wrong. `score` was: it declares `flags`
        as a positional REMAINDER, whose declared default is `None` while
        argparse hands the CLI `[]`, and `_passthrough(None)` raises TypeError.
        Only a comparison against the real parse can catch that class.
        """
        parser = argparse.ArgumentParser(prog=command.name, add_help=False)
        command.add_arguments(parser)
        required = [action for action in parser._actions if action.required]
        argv: list[str] = []
        for action in required:
            argv += [action.option_strings[0], str(_placeholder(action))]

        expected = parser.parse_args(argv)
        # The supplied values come from the real parse rather than being guessed
        # again here: `--arm` is an `append`, so argparse yields `['x']` where a
        # hand-built `'x'` would differ for reasons that say nothing about the
        # seam. What is under test is every dest NOT supplied -- the defaults
        # `arguments` fills in, which is exactly where `score` diverged.
        actual = command.arguments(**{a.dest: getattr(expected, a.dest) for a in required})
        assert vars(actual) == vars(expected)


class TestARefusalIsAValue:
    def test_a_real_command_refuses_without_exiting(self, published):
        """End-to-end: the wiring and the error channel in one call."""
        assert published.is_dir()
        with pytest.raises(CommandError, match="Run not found"):
            progress.COMMAND.invoke(run="nope", last=25)

    def test_a_cloud_config_failure_is_one_too(self):
        """So a surface catches ONE type, not a list that grows per module."""
        assert issubclass(CloudConfigError, CommandError)


class TestTheCommandLinePutsTheExitBack:
    """The CLI keeps its old behaviour; it just no longer imposes it on others."""

    def test_a_refusal_is_exit_1_and_a_message_on_stderr(self, published, capsys):
        assert published.is_dir()
        code = headless.main(["progress", "--run", "nope"])
        assert code == 1
        assert "Run not found" in capsys.readouterr().err

    def test_a_bug_still_tracebacks(self, monkeypatch, published):
        """Only CommandError is translated. A ValueError is a bug, and a
        traceback is the correct output for one."""

        def _boom(*_args, **_kwargs):
            raise ValueError("kaboom")

        monkeypatch.setattr(progress, "resolve_run_dir", _boom)
        with pytest.raises(ValueError, match="kaboom"):
            headless.main(["progress", "--run", "r"])


def test_no_command_signals_a_refusal_by_exiting_the_process():
    """The regression guard.

    `raise SystemExit` is the idiom every one of these modules used, so a new
    command copied from an old one reintroduces it silently -- it would still
    pass on the command line, and only break the surfaces that are not one.
    """
    # `rglob`, not `glob`: `commands/` is flat today, but a guard that silently
    # stops covering a package the moment someone adds a subdirectory is worse
    # than no guard, because it still reads as coverage. `flows/` is included
    # for the same reason -- it is the surface that had to catch the SystemExit.
    roots = [
        Path("src/interfaces/cli/commands"),
        Path("src/interfaces/cli/flows"),
        Path("src/interfaces/cloud"),
    ]
    offenders = [
        str(path)
        for root in roots
        for path in root.rglob("*.py")
        if "raise SystemExit" in path.read_text()
    ]
    assert offenders == [], f"raise CommandError instead: {offenders}"
