"""Sampling a running task's stacks, when an operator asks mid-job.

WHY A FILE ON THE SHARE, and not a submit flag. The environment a task runs
under is `wire.encode(spec)` over a closed `wire.KEYS`, so a dispatch-time knob
would need a wire key, a spec field, a `submit` flag and a console body field --
four contract surfaces to decide something at the wrong moment. A sampling
profiler earns its keep when you have already NOTICED something: a run whose
it/s halved at hour three, a box that is slower than its twin. The operator
drops a request beside the run, this notices within a poll, and the profile
lands on the share next to everything else.

WHY NOT `legs/`. That directory is a closed contract -- `records.REGISTRY`
keys off its suffixes and every reader goes through `read_documents` -- and a
speedscope blob is not a record. `profiles/` is its own directory for that
reason.

Nothing here may fail a task. A profiler that kills six hours of training
because ptrace was denied or the share was slow is strictly worse than no
profiler, so every entry point swallows what it catches and says so in the log.

Stdlib only, like the rest of the node wrapper: `py-spy` is invoked as a
subprocess, never imported.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

PROFILES_DIRNAME = "profiles"
REQUEST_SUFFIX = ".request"
PROFILE_SUFFIX = ".speedscope.json"
DEFAULT_SECONDS = 30
MAX_SECONDS = 600
POLL_SECONDS = 15
# Long enough that a worker between two batches still shows CPU, short enough
# that nobody notices it before their profile starts.
SETTLE_SECONDS = 2.0

# `uv run` is the child this wrapper starts; the interpreter doing the work is
# its grandchild. Profiling the shim shows an empty flamegraph.
_PYTHON_NAMES = ("python", "python3", "python3.13")


class Proc(NamedTuple):
    """What ``/proc/<pid>/stat`` is read for: the tree, and who is working."""

    comm: str
    ppid: int
    ticks: int  # utime + stime, cumulative since the process started


def _proc_stat(pid: int) -> Proc | None:
    """One pid's stat line, or None if it is gone.

    Parsed off the end rather than by splitting: a process can be named
    ``my prog) x`` and `comm` is the only field that may contain spaces or
    parentheses, so the closing one is the anchor. Indices below are relative
    to the field AFTER it, so ppid is 1 and utime/stime are 11 and 12.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    comm = raw[raw.find("(") + 1 : close]
    fields = raw[close + 2 :].split()
    if len(fields) < 13:
        return None
    try:
        return Proc(comm, int(fields[1]), int(fields[11]) + int(fields[12]))
    except ValueError:
        return None


def _process_table() -> dict[int, Proc]:
    """Every readable pid on the box, by pid."""
    try:
        pids = [int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    except OSError:
        return {}
    found = {pid: _proc_stat(pid) for pid in pids}
    return {pid: proc for pid, proc in found.items() if proc is not None}


def python_worker(root: int, settle: float = SETTLE_SECONDS) -> int | None:
    """The interpreter under ``root`` burning the most CPU RIGHT NOW.

    Ranked on a DELTA across ``settle``, not on the cumulative counter, because
    both cheaper rules picked the wrong process on a node and produced a
    plausible flamegraph of it:

    - deepest-first profiled `multiprocessing.resource_tracker`, which sits in
      `select()` and is spawned as deep as the workers;
    - cumulative CPU profiled the COORDINATOR, which had built 45M rows of
      shared arrays before forking and so led on total ticks while sitting in
      `connection._recv` waiting on the workers it had just started.

    When nothing is moving the cumulative counter decides instead: a wedged run
    is a thing worth profiling, and "no interpreter" would be the least useful
    answer to "why has it stopped".
    """
    before = _process_table()
    if not before:
        return None
    time.sleep(settle)
    after = _process_table()

    def under_root(pid: int, table: dict[int, Proc]) -> bool:
        """Whether ``pid`` sits anywhere below ``root``."""
        seen = set()
        while pid not in seen:
            if pid == root:
                return True
            seen.add(pid)
            parent = table[pid].ppid if pid in table else None
            if parent is None or parent <= 1:
                return False
            pid = parent
        return False

    running: dict[int, int] = {}
    total: dict[int, int] = {}
    for pid, proc in after.items():
        if not proc.comm.startswith(_PYTHON_NAMES) or not under_root(pid, after):
            continue
        running[pid] = proc.ticks - (before[pid].ticks if pid in before else proc.ticks)
        total[pid] = proc.ticks

    busiest = max(running, key=lambda pid: running[pid], default=None)
    if busiest is not None and running[busiest] > 0:
        return busiest
    stalled = max(total, key=lambda pid: total[pid], default=None)
    return stalled if stalled is not None and total[stalled] > 0 else None


def take_request(profile_dir: Path, task_id: str) -> int | None:
    """Seconds asked for, or None. Consumes the request so it fires once.

    An empty or unreadable request still profiles, for the default duration:
    the operator's intent is in the file EXISTING, and refusing over its
    contents would be the least helpful possible reading of `touch`.
    """
    request = profile_dir / f"{task_id}{REQUEST_SUFFIX}"
    try:
        if not request.is_file():
            return None
        body = request.read_text().strip()
    except OSError:
        return None

    with contextlib.suppress(OSError):
        request.unlink()

    try:
        seconds = int(body)
    except ValueError:
        return DEFAULT_SECONDS
    return max(1, min(seconds, MAX_SECONDS))


PTRACE_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")


def _ptrace_scope() -> str:
    """Yama's setting, which decides whether an attach is possible at all.

    ``1`` is the one that matters here and the one a stderr line does not say:
    a process may then only trace its own DESCENDANTS, and `py-spy` is started
    by the wrapper, which makes it the trainer's SIBLING. Reading the byte turns
    "exited 1" into a cause.
    """
    try:
        return PTRACE_SCOPE.read_text().strip()
    except OSError:
        return "unreadable"


def _py_spy(
    pid: int, seconds: int, destination: Path, *, native: bool
) -> subprocess.CompletedProcess[str]:
    """One `py-spy record` invocation.

    `--idle` because without it py-spy DROPS threads it reads as idle, and a
    worker inside a numba kernel is one of them: the first profile taken on a
    node came back a valid speedscope document with zero frames in it. An empty
    profile is worse than a failure -- it looks like an answer.

    `--nonblocking` is the DEFAULT and is dropped for the native attempt, which
    refuses it: "Can't get native stack traces with the --nonblocking option."
    So the native pass stops the process for each sample and the fallback does
    not -- which is the honest trade, since native frames are the only reason
    to pay anything at all here.

    `uv run --with` rather than a dependency: the nodes sync the dev group and
    have no use for a profiler in every image. By the time any child is running
    `uv sync` has completed, so the tool resolves from a warm cache.
    """
    argv = [
        "uv",
        "run",
        "--with",
        "py-spy",
        "py-spy",
        "record",
        "--pid",
        str(pid),
        "--duration",
        str(seconds),
        "--format",
        "speedscope",
        "--output",
        str(destination),
        "--idle",
    ]
    argv.append("--native" if native else "--nonblocking")
    return subprocess.run(argv, capture_output=True, text=True, timeout=seconds + 180, check=False)


def record(pid: int, seconds: int, destination: Path, log: Callable[[str], None]) -> bool:
    """Sample ``pid`` for ``seconds`` into ``destination``. Never raises.

    `--native` first because the interesting frames are inside numba kernels and
    a pure-Python profile of this workload is mostly one line of traversal. It
    is also the part most likely to be unavailable, so a failure falls back to
    the plain profile rather than returning nothing: the two outcomes are
    different answers -- a build without native support, versus an attach the
    box will not permit at all.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        done = _py_spy(pid, seconds, destination, native=True)
        if done.returncode != 0:
            # BOTH streams. `uv run --with` writes its own download chatter to
            # stderr, which is what the first failure on a node reported --
            # py-spy's actual complaint was nowhere in the log.
            log(f"profile: --native on pid {pid} exited {done.returncode}: {_said(done)}")
            log(f"profile: yama ptrace_scope={_ptrace_scope()} (1 = descendants only)")
            done = _py_spy(pid, seconds, destination, native=False)
    except (OSError, subprocess.SubprocessError) as error:
        log(f"profile: could not run py-spy ({error})")
        return False

    if done.returncode != 0:
        log(f"profile: py-spy on pid {pid} exited {done.returncode}: {_said(done)}")
        return False

    log(f"profile: {seconds}s of pid {pid} -> {destination.name}")
    return True


def _said(done: subprocess.CompletedProcess[str]) -> str:
    """Both streams, trimmed. Which one carries the reason is not knowable."""
    both = f"{done.stdout.strip()} {done.stderr.strip()}".strip()
    return both[-600:] if len(both) > 600 else both


def watch(
    root: int,
    profile_dir: Path,
    task_id: str,
    log: Callable[[str], None],
    stop: threading.Event,
) -> None:
    """Poll for a request beside the run and serve it. Never raises.

    One profile per request, and the request is consumed before the recording
    starts -- a slow share or a refused ptrace must not leave a file that
    re-profiles on every poll for the rest of a six-hour task.
    """
    attempt = os.environ.get("AZ_BATCH_TASK_ATTEMPT", "0")
    served = 0
    while not stop.wait(POLL_SECONDS):
        with contextlib.suppress(Exception):
            seconds = take_request(profile_dir, task_id)
            if seconds is None:
                continue
            pid = python_worker(root)
            if pid is None:
                log("profile: asked for, but no interpreter is running under this task")
                continue
            served += 1
            record(
                pid,
                seconds,
                profile_dir / f"{task_id}.{attempt}.{served}{PROFILE_SUFFIX}",
                log,
            )


def watcher(
    root: int, profile_dir: Path, task_id: str, log: Callable[[str], None]
) -> tuple[threading.Thread, threading.Event]:
    """A started daemon thread serving profile requests, and its stop flag.

    It announces the path it is watching because silence is otherwise
    unreadable: with no line here, a profile that never appears cannot be told
    apart from a watcher that never armed.
    """
    log(f"profile: watching {profile_dir}/{task_id}{REQUEST_SUFFIX} (seconds in the body)")
    stop = threading.Event()
    thread = threading.Thread(
        target=watch, args=(root, profile_dir, task_id, log, stop), name="profile", daemon=True
    )
    thread.start()
    return thread, stop
