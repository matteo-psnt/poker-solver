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
import ast
from typing import TYPE_CHECKING, Any

import pytest

from src.interfaces.cli import headless
from src.interfaces.cloud.config import CloudConfigError
from src.interfaces.commands import load_all, progress
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared import repo

if TYPE_CHECKING:
    from pathlib import Path


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
        assert isinstance(payload, dict)
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


# Where ending the process IS the answer. Naming the two sites here rather than
# widening the guard keeps the next one a decision.
EXITS_ON_PURPOSE: dict[str, str] = {
    "interfaces/cli/headless.py": (
        "The `__main__` guard. This is the one place a refusal becomes an exit "
        "code, which is the whole point of the seam -- everything above it "
        "returns an int."
    ),
    "interfaces/commands/blueprint_serve.py": (
        "Not a refusal: the exit CODE is how the server tells its systemd unit "
        "why it stopped. Idle expiry must switch the box off and every other "
        "stop must not, and they are indistinguishable by then. Cost 62 hours "
        "of a box restarting itself -- see `idle.IDLE_EXIT_CODE`."
    ),
}


def _system_exit_lines(path: Path) -> list[int]:
    """Lines that actually raise ``SystemExit``, parsed rather than grepped.

    The substring form counted this package's own ``__init__`` docstring, which
    names the idiom in order to say it was removed. A guard that fires on prose
    about itself trains people to widen it.
    """
    return [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Raise)
        and isinstance(exc := node.exc, ast.Call)
        and isinstance(exc.func, ast.Name)
        and exc.func.id == "SystemExit"
    ]


def test_no_command_signals_a_refusal_by_exiting_the_process():
    """The regression guard.

    `raise SystemExit` is the idiom every one of these modules used, so a new
    command copied from an old one reintroduces it silently -- it would still
    pass on the command line, and only break the surfaces that are not one.
    """
    # `rglob`, not `glob`: `commands/` is flat today, but a guard that silently
    # stops covering a package the moment someone adds a subdirectory is worse
    # than no guard, because it still reads as coverage.
    #
    # Which is what happened to this one. It named `cli/commands` and
    # `cli/flows`, and both moved out from under it -- the commands to
    # `interfaces/commands` when the console became a second caller, the flows
    # to deletion with the interactive menu. Two of its three roots had not
    # existed for weeks, `rglob` on a missing directory yields nothing rather
    # than raising, and the test went on passing over `cloud/` alone while
    # reading as cover for every command module. Anchored to the repo now, so a
    # move relocates the code out of a root that still exists.
    roots = [
        repo.SRC / "interfaces" / "commands",
        repo.SRC / "interfaces" / "cli",
        repo.SRC / "interfaces" / "web",
        repo.SRC / "interfaces" / "cloud",
    ]
    missing = [str(root) for root in roots if not root.is_dir()]
    assert missing == [], f"this guard names directories that do not exist: {missing}"

    offenders = [
        f"{path.relative_to(repo.ROOT)}:{line}"
        for root in roots
        for path in root.rglob("*.py")
        for line in _system_exit_lines(path)
        if path.relative_to(repo.SRC).as_posix() not in EXITS_ON_PURPOSE
    ]
    assert offenders == [], (
        "raise CommandError instead — a refusal must be a value, or every "
        f"surface that is not a command line inherits the exit: {offenders}"
    )


def test_only_the_errors_module_names_the_azure_sdks_exceptions():
    """The ladder is written once, or it is written once per surface.

    ``status`` and the console both have to survive an expired ``az login``, and
    both carried the same three-arm ``except`` -- differing only in whether the
    result became an HTTP status or a dict field. Nothing failed when they
    agreed; the cost was that a third surface would write it a third time, and
    the cost of getting one arm wrong is a whole screen blanking where two
    panels should have.

    ``errors.attempt`` is that ladder. This is the guard that keeps it the only
    one: a new surface copying the old idiom would pass every other test here.
    """
    # BOTH names. An earlier draft checked only `ClientAuthenticationError`,
    # which leaves a surface catching `HttpResponseError` alone -- an
    # unreachable endpoint, the other half of the pair -- passing a guard whose
    # docstring claims to cover the ladder.
    names = ("ClientAuthenticationError", "HttpResponseError")
    allowed = {repo.SRC / "interfaces" / "errors.py"}
    offenders = [
        f"{path.relative_to(repo.ROOT)} ({name})"
        for path in (repo.SRC / "interfaces").rglob("*.py")
        if path not in allowed
        for name in names
        if name in path.read_text()
    ]
    assert offenders == [], (
        "these catch the Azure SDK's exceptions directly — go through "
        "`errors.attempt`, which classifies them once: " + ", ".join(offenders)
    )
