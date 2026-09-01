"""Handing this task's own records to a process that can store them.

The wrapper writes leg documents to the share and cannot write them to the
database: `infra/run_task.py` runs on the pool's bootstrap interpreter, not the
venv `uv sync` builds, so the driver is not on this process's path -- before or
after that sync -- and `shared` may not import an adapter in any case. Crossing
into a process that HAS the driver is not a workaround for those rules; it is
the only way to satisfy them, and it is how the wrapper already runs the task.

NEVER FATAL, and THROTTLED. The share holds the record and this is the copy, so
a task must not die making it -- and must not spend meaningful wall clock on it
either, which is the half that bit first: see :data:`MIN_INTERVAL_SECONDS`.
"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING

from src.shared.cloudtask import task_log
from src.shared.cloudtask.node.process import terminate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# MEASURED on a node: the first `uv run` after a fresh `uv sync` takes over a
# minute -- the project install plus a cold import off an empty page cache --
# while every later one is quick. A 60s ceiling timed that first call out.
TIMEOUT_SECONDS = 240

# The watcher publishes progress every 15s, and mirroring on each of them is
# what turned a 20.5s training run into a task that finished seven minutes
# later: the cold calls each burned the full ceiling and stacked up on the
# watcher thread, which `stop()` then joins with a 120s grace before sampling
# once more. A leg row is worth having within a couple of minutes and is worth
# nothing to the fifteen-second cadence, so the CADENCE IS DECIDED HERE rather
# than by whoever calls it.
MIN_INTERVAL_SECONDS = 120

_last_run = 0.0


def publish(
    share: Path,
    task_id: str,
    *,
    cwd: Path,
    log: Callable[[str], None] | None = None,
    force: bool = False,
) -> None:
    """Mirror this task's leg documents, best effort and at most every 120s.

    `force` is for the EXIT path, where the throttle would drop the one call
    that matters: the record has just reached its terminal state and there is no
    later tick to carry it.
    """
    global _last_run
    now = time.monotonic()
    if not force and now - _last_run < MIN_INTERVAL_SECONDS:
        return
    _last_run = now
    argv = [
        "uv",
        "run",
        "poker-solver",
        "mirror-legs",
        "--task",
        task_id,
        "--legs-dir",
        str(task_log.tasks_dir(share)),
    ]
    try:
        # Its OWN SESSION, so a timeout can signal the whole tree: `uv run`
        # spawns a python grandchild and `terminate()` reaches only `uv`,
        # leaving that python running -- the reason `run_guarded` does the same,
        # and `terminate` below is ITS teardown rather than a second one.
        # DEVNULL because there is nothing here worth reading and a pipe a
        # grandchild still owns is one more thing that can block.
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        # Expected before `uv sync` has run, where there is no `uv run` to call.
        if log:
            log(f"mirror-legs unavailable: {type(exc).__name__}")
        return

    try:
        code = process.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        terminate(process)
        if log:
            log(f"mirror-legs timed out after {TIMEOUT_SECONDS}s")
        return
    if code != 0 and log:
        log(f"mirror-legs rc={code}")
