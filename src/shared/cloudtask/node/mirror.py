"""Handing this task's own records to a process that can store them.

The wrapper writes leg documents to the share and cannot write them to the
database: `infra/run_task.py` runs on the pool's bootstrap interpreter, not the
venv `uv sync` builds, so the driver is not on this process's path -- before or
after that sync -- and `shared` may not import an adapter in any case. Crossing
into a process that HAS the driver is not a workaround for those rules; it is
the only way to satisfy them, and it is how the wrapper already runs the task.

NEVER FATAL. The share holds the record and this is the copy: a task must not
die making it. The CADENCE is not decided here -- the watcher's coarse tick
already exists for work that belongs on the slow path, and mirroring on the 15s
progress cadence instead put three and a half minutes on the end of a task whose
training took twenty seconds.
"""

from __future__ import annotations

import subprocess
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


def publish(
    share: Path, task_id: str, *, cwd: Path, log: Callable[[str], None] | None = None
) -> None:
    """Mirror this task's leg documents, best effort.

    Reads the task's CURRENT records rather than a delta, so a call that fails
    is repaired by the next one instead of losing a document forever.
    """
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
