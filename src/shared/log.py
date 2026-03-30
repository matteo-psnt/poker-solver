"""Bare-format logging for the library layers (pipeline/engine).

Library code must not ``print()``: it pollutes stdout that machine consumers
(``poker-solver-run --json``) need clean. Modules log via
``logging.getLogger(__name__)`` instead, and every process entrypoint calls
``configure_logging()`` once.

Records go to stderr, keeping stdout for payloads. That split is what makes
``poker-solver-run --json | jq`` work.

TWO READERS, TWO FORMATS. Interactively the bare message is right. Captured --
a cloud leg's log, a ``tee``, CI -- it is not: every severity renders
identically, so a 2 MB leg log cannot be grepped for the failure it exists to
explain. The format keys on ``stderr_is_terminal()``, the same predicate that
gates progress bars. Interactive output is unchanged; captured output gains a
UTC timestamp, a level and the logger name.

SPAWNED PROCESSES inherit no handler on the ``src`` logger, so a child that
skips ``configure_logging()`` falls through to ``logging.lastResort`` at WARNING
and every ``logger.info`` it makes is dropped. Every worker entrypoint calls it
first, even when the worker body never logs -- the libraries it pulls in do.

LEVEL: ``POKER_SOLVER_LOG_LEVEL`` beats the caller's level, which beats INFO.
The environment because it is the only channel that survives a spawn boundary
without threading through every worker signature; ``--log-level`` sets it.

The two stdout redirections in the tree (``cli/headless.py`` under ``--json``,
the LBR pool initializer) are for third-party writers this cannot reach, and are
not removable.
"""

import logging
import os
import sys
import time
from typing import TextIO

LEVEL_ENV_VAR = "POKER_SOLVER_LOG_LEVEL"

# Python's WARNING and CRITICAL are 7 and 8 chars, which would ragged the column.
_SHORT_LEVEL = {"WARNING": "WARN", "CRITICAL": "CRIT"}

# The median logger name once ``src.`` is stripped; the longest is 49. Padding to
# the max would indent the whole log for a handful of modules.
_NAME_WIDTH = 34


class _DynamicStderrHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stderr`` at emit time.

    A plain ``StreamHandler(sys.stderr)`` binds the stream object once, so
    later redirections (pytest ``capsys``, worker-side stream swaps) silently
    write to a dead buffer. Resolving per record follows the current stderr.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, value: TextIO) -> None:
        """Ignored: resolved on every read, since pytest swaps sys.stderr
        after the handler is built."""


class _AdaptiveFormatter(logging.Formatter):
    """Bare message on a terminal; timestamped and levelled when captured.

    Resolved per record, not at install: the same process can be interactive and
    then have its stderr swapped (spawned worker, pytest capture).
    """

    def __init__(self) -> None:
        super().__init__(
            f"%(asctime)s %(levelname)-5s %(name)-{_NAME_WIDTH}s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        self.converter = time.gmtime  # legs run in whatever region had capacity
        self._bare = logging.Formatter("%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if stderr_is_terminal():
            return self._bare.format(record)
        # Mutate and restore rather than copy: the record is shared with any
        # other handler on the chain, and makeLogRecord drops the exc_text cache.
        name, levelname = record.name, record.levelname
        record.name = name.removeprefix("src.")
        record.levelname = _SHORT_LEVEL.get(levelname, levelname)
        try:
            return super().format(record)
        finally:
            record.name, record.levelname = name, levelname


def _numeric_level(level: int | str) -> int:
    """Numeric value of one level, raising on an unrecognised name.

    Raising rather than defaulting: a typo'd ``POKER_SOLVER_LOG_LEVEL=DEGUB``
    that silently kept INFO looks exactly like the override not working.
    """
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"unknown log level {level!r}")
    return resolved


def resolve_level(level: int | str = logging.INFO) -> int:
    """Resolve a level to its numeric value, letting the environment win."""
    return _numeric_level(os.environ.get(LEVEL_ENV_VAR) or level)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install the adaptive stderr handler on the ``src`` package logger.

    Idempotent: repeat calls only adjust the level. Handlers attach to the
    package root (not the global root) so third-party loggers keep their own
    configuration, and propagation is cut so nothing double-prints.
    """
    package_logger = logging.getLogger("src")
    if not package_logger.handlers:
        handler = _DynamicStderrHandler()
        handler.setFormatter(_AdaptiveFormatter())
        package_logger.addHandler(handler)
        package_logger.propagate = False
    package_logger.setLevel(resolve_level(level))


def pin_level_for_children(level: str) -> None:
    """Publish ``level`` into the environment so spawned workers inherit it.

    Called from the CLI seam for ``--log-level``; without it a worker built from
    ``config.system.log_level`` would disagree with the coordinator.
    """
    # Validate the ARGUMENT: resolve_level consults the environment first, so an
    # already-set variable would mask the typo being published over it.
    _numeric_level(level)
    os.environ[LEVEL_ENV_VAR] = level.upper()


def stderr_is_terminal() -> bool:
    """Is a human watching this scroll past, or is it being captured?

    The one predicate behind both the log format and progress bars. Resolved per
    call because stderr can be swapped after configuration.
    """
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        # Closed or exotic stream: assume captured. A structured line in a
        # terminal is untidy; a bare line in a log is unsearchable.
        return False


def progress_bars_enabled() -> bool:
    """Whether tqdm bars should render.

    A bar repaints its whole line on every update, which is invisible
    interactively and catastrophic in a captured log: a multi-hour cloud leg
    once wrote a stderr file too large to download, destroying the one artifact
    needed to diagnose a stalled run.
    """
    return stderr_is_terminal()
