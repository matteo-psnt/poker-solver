"""What every command invocation cost, recorded where caches live.

One row per command that ran, so "what is slow, and how often does it fail" is a
question with an answer rather than an impression. `poker-solver activity` reads
it.

**Laptop-local and disposable.** It goes under
:func:`~src.shared.cache.cache_root`, never the share: the share has no atomic
append, a document per invocation would outgrow `legs/` in hours, and every
write there is an SMB round trip -- so instrumenting round-trip cost would ADD
one to every command.

**Never the reason a command fails.** Every write is best-effort: a full disk, a
read-only cache directory, a value that will not serialise -- none of them may
turn a working command into a failing one.

Best-effort is not silent, though. A cache directory that cannot be written
stops the log FOREVER, and `activity` then reports "no commands recorded yet"
while commands are plainly running -- a wrong answer with a plausible
explanation. So the first failure is logged at WARNING and the rest at DEBUG.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.interfaces.errors import CommandError
from src.shared import cache, records

"""``POKER_SOLVER_TELEMETRY``. Named for the subject rather than for the switch:
the value that turns it OFF is `0`/`off`/`false`/`no`, so an `ENV_DISABLE=0`
would have read as "disable = no" and meant the opposite of what it does."""
ENV_VAR = "POKER_SOLVER_TELEMETRY"

ARTIFACT = "telemetry/invocations.jsonl"

"""Rotate at a size rather than a row count or an age.

Rows differ by an order of magnitude in width -- a `submit` carries nine
arguments, a `pool-status` carries none -- so a row cap bounds the wrong thing.
An age cap would need the file read and rewritten on some schedule, which is a
lot of machinery for a file nobody is obliged to keep.

One previous generation is kept, and :func:`logs` is what makes that true --
`activity` reads BOTH, or the rotation would silently do the thing this policy
claims to prevent: the moment the log crossed the cap, every window would report
on a near-empty file. Two generations would double the disk for a marginal gain
in history; zero would lose everything at once, including the minute before
whatever someone is currently investigating.

Rotation is racy between processes -- two commands can both see the file over
the cap and both rename, and the second clobbers what the first saved. Left
that way ON PURPOSE: a lock file shared across every invocation of every command
is real machinery, and what it would protect is a bounded number of rows in a
file that is disposable by definition. It is a cost worth naming rather than
paying for.
"""
MAX_BYTES = 8 * 1024 * 1024

"""Which surface asked.

A ContextVar rather than a parameter: the argument list of ``execute`` belongs
to the COMMAND, and adding a `surface=` to it would put a field there that no
command declares and that could collide with one that did. Each entry point sets
it once -- `headless` to `cli`, the console's `answer` to `console` -- and
anything that sets neither is honestly reported as unknown rather than being
quietly filed as the command line.
"""
_SURFACE: ContextVar[str] = ContextVar("surface", default="unknown")

logger = logging.getLogger(__name__)

"""Whether the "this is not working" line has already been said.

Process-global rather than per-path: the point is to say it ONCE, and a
per-path map would repeat it for a rotation."""
_complained = False


def log_path() -> Path:
    """Where the rows go. Not created here; the writer creates its parent."""
    return cache.cache_dir("telemetry") / "invocations.jsonl"


def logs() -> list[Path]:
    """Every generation a reader should fold in, OLDEST first.

    The rotated file is not an archive to be dug out by hand: keeping it is only
    worth anything if the reader reads it, and without this `activity` answered
    from the live file alone -- so crossing the 8 MB cap would have emptied
    every window it reports, which is exactly the loss rotation exists to avoid.
    """
    current = log_path()
    return [path for path in (current.with_suffix(".jsonl.1"), current) if path.is_file()]


def enabled() -> bool:
    """Off only when asked. Recording is the default, or nothing accumulates.

    The test suite turns it off in `conftest`, for two reasons that are both
    about honesty rather than speed: a test run would otherwise write thousands
    of rows into the developer's real cache, and those rows describe commands
    that never ran against anything.
    """
    return os.environ.get(ENV_VAR, "").strip().lower() not in {"0", "off", "false", "no"}


@contextmanager
def surface(name: str) -> Iterator[None]:
    """Attribute everything invoked inside this block to one surface."""
    token = _SURFACE.set(name)
    try:
        yield
    finally:
        _SURFACE.reset(token)


@contextmanager
def observe(command: str, asked: dict[str, Any]) -> Iterator[None]:
    """Time one command and record how it ended. Re-raises whatever it caught.

    ``outcome`` distinguishes three things a reader acts on differently:
    ``ok``, ``refusal`` (a :class:`CommandError` -- the command understood and
    the answer was no, which is not a fault), and ``error`` (anything else).

    Azure's failures are deliberately NOT a fourth arm. That ladder is declared
    once, in :func:`~src.interfaces.errors.attempt`, and a guard test fails if
    anything under `interfaces/` names the SDK's exception types again -- a text
    scan, deliberately blunt, and its value is that it has no exceptions. So
    this records the exception's TYPE NAME as data instead. `activity` groups by
    it, which makes a run of expired-credential failures just as visible without
    this module holding a second opinion about what they mean.

    ``monotonic`` for the duration, ``now()`` for the timestamp: a wall clock
    that steps backwards over an NTP correction would otherwise produce a
    negative duration, which reads as a bug in whatever is being measured.
    """
    started = time.monotonic()
    at = datetime.now(UTC).isoformat()
    outcome, error_type, message = "ok", "", ""
    try:
        yield
    except CommandError as error:
        outcome, error_type, message = "refusal", type(error).__name__, str(error)
        raise
    except BaseException as error:
        outcome, error_type, message = "error", type(error).__name__, str(error)
        raise
    finally:
        _write(
            {
                "at": at,
                "command": command,
                "surface": _SURFACE.get(),
                "seconds": round(time.monotonic() - started, 4),
                "outcome": outcome,
                "error_type": error_type,
                # Truncated: a traceback-length message in a log meant to be
                # scanned by the thousand crowds out the row it sits in, and the
                # first line is what identifies the failure.
                "error": message[:300],
                "asked": asked,
            }
        )


def asked_for(
    add_arguments: Any, args: argparse.Namespace, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The arguments that differ from their defaults -- what the caller ASKED for.

    Recording the whole Namespace would be mostly noise: a command with twelve
    flags is invoked with one or two, and the other ten are the same values in
    every row. What is left is the part that varies, which is the part worth
    grouping by -- which run was slow, which config, whether the expensive flag
    was set.

    ``defaults`` is accepted so a caller that already built the parser does not
    build it twice; without it the parser is constructed here.
    """
    if defaults is None:
        parser = argparse.ArgumentParser(add_help=False)
        add_arguments(parser)
        defaults = {action.dest: action.default for action in parser._actions}  # noqa: SLF001
    return {
        key: value
        for key, value in vars(args).items()
        # `log_level` and `json` are the command line's own flags rather than
        # the command's, and they say nothing about what was asked for.
        if key in defaults and value != defaults[key] and key not in {"json", "log_level"}
    }


def _write(row: dict[str, Any]) -> None:
    """Append one row, or silently do nothing. Never raises."""
    if not enabled():
        return
    try:
        path = log_path()
        _rotate(path)
        records.append_log(path, row, records.REGISTRY[ARTIFACT])
    except Exception as error:  # noqa: BLE001 — a bystander must not fail the work
        _complain(error)


def _complain(error: BaseException) -> None:
    """Say once that recording is not working, then stop saying it."""
    global _complained
    if _complained:
        logger.debug("telemetry write failed: %s", error)
        return
    _complained = True
    logger.warning(
        "Command telemetry is not being recorded (%s: %s). `poker-solver activity` "
        "will look empty. This does not affect any command; set %s=0 to silence it.",
        type(error).__name__,
        error,
        ENV_VAR,
    )


def _rotate(path: Path) -> None:
    """Move the log aside once it is large, keeping one generation.

    Checked before the append rather than after, so the cap is a bound on what
    exists rather than on what existed a moment ago.
    """
    if not path.is_file() or path.stat().st_size < MAX_BYTES:
        return
    path.replace(path.with_suffix(".jsonl.1"))
