"""Handing this task's own records to a process that can store them.

The wrapper writes leg documents to the share and cannot write them to the
database: `infra/run_task.py` runs on the pool's bootstrap interpreter, not the
venv `uv sync` builds, so the driver is not on this process's path -- before or
after that sync -- and `shared` may not import an adapter in any case. Crossing
into a process that HAS the driver is not a workaround for those rules; it is
the only way to satisfy them, and it is how the wrapper already runs the task.

NEVER FATAL, at every call site. The share holds the record; this is the copy.
A task must not die because the copy could not be made, and the node's log
saying so is the whole of the failure handling this deserves.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from src.shared.cloudtask import task_log

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Generous against a cold connection, short against a hang. A first connect from
# a node costs a TLS handshake and SCRAM auth; a wedged one must not outlive the
# 120s between progress ticks, or two of these would overlap.
TIMEOUT_SECONDS = 60


def publish(
    share: Path, task_id: str, *, cwd: Path, log: Callable[[str], None] | None = None
) -> None:
    """Mirror this task's leg documents, best effort.

    Called from the WATCHER THREAD during a task and from the exit path after
    it, so the seconds this takes are not seconds the work is not running.
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
        done = subprocess.run(
            argv, cwd=cwd, timeout=TIMEOUT_SECONDS, capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Expected before `uv sync` has run, where there is no `uv run` to call.
        if log:
            log(f"mirror-legs unavailable: {type(exc).__name__}")
        return
    if done.returncode != 0 and log:
        log(f"mirror-legs rc={done.returncode}: {(done.stderr or '').strip()[:200]}")
