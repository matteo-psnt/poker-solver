"""Running a child process under a deadline, and keeping what it said.

The two things a Batch task cannot afford to get wrong about a subprocess: that
it might never exit, and that its output lives on a node the pool destroys
minutes later. Each rule below replaced a shell construct that had already
failed in production, and each is argued where it happens -- the signal handler
raises rather than flags, the tee reads chunks because tqdm emits no newlines,
and a child killed by a signal has its shell code restored so 137 keeps meaning
what ``poker-solver tasks`` reads it as.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

"""`timeout`'s convention, kept because `task_log` maps it to a distinct cause
and a wrong terminal cause is permanent -- it suppresses reconciliation."""
EXIT_TIMEOUT = 124

"""How long the trainer gets to flush after SIGTERM before SIGKILL. It exists
because the case that motivated the guard was a process that ignored TERM."""
GRACE_SECONDS = 120

"""The interesting part of a task log is always the end, and a multi-hour tqdm
stream is mostly progress-bar repaints that cost more to copy than they
inform."""
PUBLISHED_LOG_BYTES = 2_000_000

CHUNK = 65536


class Killed(BaseException):
    """A signal reached the wrapper. ``BaseException`` so no broad ``except``
    can swallow the one event the exit record exists to report."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class TaskLogger:
    """Tees everything to node-local disk, then to the share on demand.

    Batch keeps a task's stdout ON THE NODE and the pool drains within minutes
    of a task ending, so anything only echoed is gone for exactly the tasks
    worth reading later -- which is what happened to a 30M task that died at
    ~720k iterations, leaving ``exit 1`` and nothing else. Node-local first
    because the training stream is chatty and writing every line straight to
    SMB would put the task's throughput at the mercy of the share.
    """

    def __init__(self, path: Path, share: Path) -> None:
        self.path = path
        self.share = share
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("ab")
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        self.write(f"[run_task {stamp}] {message}\n".encode())

    def write(self, chunk: bytes) -> None:
        with self._lock:
            self._handle.write(chunk)
            self._handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    def publish(self) -> None:
        """Copied on every publish, not only at exit, so a task later killed
        outright still leaves its log behind."""
        task = os.environ.get("AZ_BATCH_TASK_ID", "task")
        destination = self.share / "logs" / f"{task}.log"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("rb") as source:
                size = source.seek(0, os.SEEK_END)
                source.seek(max(0, size - PUBLISHED_LOG_BYTES))
                destination.write_bytes(source.read())
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def run_guarded(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
    log: TaskLogger,
    stdout_to: Path | None = None,
) -> int:
    """Run a subprocess under a wall-clock ceiling, teeing its output.

    The task-level ``maxWallClockTime`` (P1D) is not a backstop for a hang: it
    is longer than any task is meant to run, so a wedged process bills a full
    node-day before Batch acts. One task proved this -- training died, the
    process could not exit, and the task stayed ``running`` indefinitely.

    ``stdout_to`` captures stdout to a file instead of teeing it, for the one
    command whose stdout is a JSON payload rather than a log.
    """
    sink = stdout_to.open("wb") if stdout_to else None
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=sink if sink else subprocess.PIPE,
            stderr=subprocess.PIPE if sink else subprocess.STDOUT,
            close_fds=True,
            # Its OWN group, so the guard can signal the whole tree.
            # `terminate()` reaches only `uv`; the trainer and its 16 workers
            # are grandchildren, and a deadline used to return 124 with them
            # still running, holding the /dev/shm segments that killed the NEXT
            # task. Batch cleans up by cgroup, so a new session does not escape.
            start_new_session=True,
        )
    except OSError as error:
        log(f"FATAL could not start {argv[0]}: {error}")
        if sink:
            sink.close()
        return 1

    pump = threading.Thread(
        target=_pump,
        args=(process.stderr if sink else process.stdout, log),
        name="tee",
        daemon=True,
    )
    pump.start()
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            log(f"TIMEOUT after {timeout}s -- guard fired; published rungs are on the share")
            _terminate(process)
    except Killed:
        # The wrapper itself was signalled. Take the child down with it, then
        # let the exception carry the cause to the exit record.
        _terminate(process)
        raise
    finally:
        pump.join(timeout=GRACE_SECONDS)
        if sink:
            sink.close()

    if timed_out:
        return EXIT_TIMEOUT
    # A child killed by a signal reports a negative code. Restore the shell's
    # 128+n so 137 keeps meaning what `tasks` reads it as: SIGKILL from outside,
    # which on a training node is the OOM killer.
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def _pump(stream: IO[bytes] | None, log: TaskLogger) -> None:
    """Chunked, never line-buffered: tqdm emits carriage returns, not newlines.

    ``read1`` where the pipe offers it: it returns as soon as ANY bytes are
    available, where ``read`` waits for a full chunk -- which on a stream that
    repaints one progress bar is a tee that goes quiet for minutes.
    """
    if stream is None:
        return
    read: Callable[[int], bytes] = getattr(stream, "read1", stream.read)
    try:
        while chunk := read(CHUNK):
            log.write(chunk)
    except (OSError, ValueError):
        return


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """TERM the whole group so the trainer's handlers can flush, KILL if it
    will not go -- the case that motivated the guard ignored TERM."""
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait()


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
    """Falls back to the child alone if the group is already gone."""
    try:
        os.killpg(os.getpgid(process.pid), signum)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            process.send_signal(signum)
