"""Result-queue gathering shared by coordinator broadcast operations."""

from __future__ import annotations

import logging
import os
import queue
import signal
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from .manager import SharedArrayWorkerManager

# How often to look up from the queue to check whether the workers are still
# alive. Short enough that a death is reported in seconds rather than after the
# full inter-message timeout, long enough to cost nothing.
LIVENESS_POLL_S = 2.0


def _describe_exit(exitcode: int) -> str:
    """Human-readable cause for a worker's exit code.

    A negative code is ``-signum``. SIGKILL is called out by name because it is
    almost always the kernel OOM killer -- the one cause that leaves no traceback
    anywhere, and so the one worth naming rather than leaving as a bare number.
    """
    if exitcode >= 0:
        return f"exited with status {exitcode}"
    try:
        name = signal.Signals(-exitcode).name
    except ValueError:
        return f"killed by signal {-exitcode}"
    if name == "SIGKILL":
        return "killed by SIGKILL (no traceback is possible; usually the OOM killer)"
    return f"killed by {name}"


def _dead_workers(processes: Sequence[BaseProcess]) -> list[str]:
    """Descriptions of every worker that has already exited."""
    return [
        f"worker pid={p.pid} {_describe_exit(p.exitcode)}"
        for p in processes
        if p.exitcode is not None
    ]


def _worker_status(processes: Sequence[BaseProcess]) -> str:
    alive = sum(1 for p in processes if p.exitcode is None)
    dead = _dead_workers(processes)
    parts = [f"{alive}/{len(processes)} workers still alive"]
    if dead:
        parts.append("; ".join(dead))
    return " | ".join(parts)


def gather_worker_results(
    manager: SharedArrayWorkerManager,
    accept: Callable[[dict[str, object]], bool],
    expected: int,
    timeout: float,
    description: str,
    verbose: bool = False,
) -> tuple[list[dict[str, object]], bool]:
    """
    Collect `expected` accepted results from the manager's result queue.

    Loops until `expected` accepted messages arrive; unrecognized messages are
    discarded without consuming a completion slot, so a stray late result can
    never make a broadcast operation report success while acks are missing.

    ``timeout`` bounds the wait BETWEEN messages, not the whole gather, so a slow
    batch is not penalised for having many workers.

    A dead worker and a slow one used to be indistinguishable: this blocked for
    the full timeout and then reported only a count, so a leg whose workers had
    been killed spent ten minutes waiting and produced no evidence of why. The
    queue is now polled, worker liveness is checked between polls, and a death
    aborts immediately naming the pid and signal.

    Returns (results, interrupted); `interrupted` is True if a KeyboardInterrupt
    arrived while waiting (the gather keeps waiting so workers finish cleanly).
    """
    results: list[dict[str, object]] = []
    interrupted = False
    deadline = time.monotonic() + timeout
    while len(results) < expected:
        remaining = deadline - time.monotonic()
        try:
            raw_result = manager.result_queue.get(timeout=min(LIVENESS_POLL_S, max(0.0, remaining)))
        except queue.Empty:
            # Death first: it is the specific, actionable answer, and reporting a
            # timeout when the real event was a SIGKILL sends the next reader
            # looking for a performance problem that does not exist.
            dead = _dead_workers(manager.processes)
            if dead:
                raise RuntimeError(
                    f"{len(dead)} worker(s) died while waiting for {description} "
                    f"({len(results)}/{expected} received): " + "; ".join(dead)
                )
            if remaining <= 0:
                raise RuntimeError(
                    f"Timeout waiting for {description} after {timeout:.0f}s with no "
                    f"message ({len(results)}/{expected} received). "
                    f"{_worker_status(manager.processes)}. "
                    f"Master rss={_rss_mb()} MB."
                )
            continue
        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                logger.info(
                    f"⚠️  Interrupt received; waiting for {description}...",
                )
            continue
        deadline = time.monotonic() + timeout
        if not isinstance(raw_result, dict):
            if verbose:
                logger.info(f"[Master] Ignoring unexpected non-dict result: {raw_result}")
            continue
        result = cast(dict[str, object], raw_result)
        if accept(result):
            results.append(result)
        elif verbose:
            logger.info(f"[Master] Ignoring unexpected result: {result}")
    return results, interrupted


def _rss_mb() -> str:
    """Resident set size of this process, or '?' where it is not readable.

    Reported alongside a stall because memory pressure is the leading explanation
    for workers going silent, and the number is worthless after the fact.
    """
    try:
        with open(f"/proc/{os.getpid()}/statm") as handle:
            pages = int(handle.read().split()[1])
        return str(pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024))
    except (OSError, ValueError, IndexError):
        return "?"
