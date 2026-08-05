"""What a task IS, separated from how it gets dispatched.

This module is deliberately free of the Azure SDK. Everything here is a pure
function of its arguments -- which is the point: the parts of a submission that
can be *wrong* (an iteration target that is relative instead of absolute, an
override that loses half its value, a task id that Batch rejects) are decided
here, where a test can look at them, rather than inside a string being handed
to a service.

The node contract
-----------------
``infra/run_task.py`` reads every value below out of the task's environment --
see :mod:`src.shared.node.plan`, which is the same contract from the node's
end and is pinned against this one by ``tests/shared/node/test_plan.py``. Two
of them are JSON-encoded rather than passed raw, and the reason is not the one
the old shell had. The ``az`` CLI needed hex because
``--environment-settings`` parses ``KEY=VALUE`` and a config override's value
contains ``=``; the SDK has no such problem. JSON survives here because
overrides and eval flags are *lists whose elements may contain spaces*, and the
old space-joined form silently split them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.shared import tasks
from src.shared.tasks import BadTaskError, TaskName

DEFAULT_TIMEOUT = "6h"
DEFAULT_CHECKPOINT_EVERY = 1_000_000

"""The node's interpreter
------------------------
NOT the OS python3, which is 3.10 on the pinned 22.04 image. The start task
installs this one and links it into a system bin directory, so the wrapper runs
a plain interpreter at a fixed absolute path -- no uv at task time.

That indirection is the point. `uv run` here needed uv to resolve a managed
interpreter, and uv resolves it under HOME: the start task runs as an elevated
POOL-scoped auto-user and a task runs as a different, non-elevated one whose
HOME is its own working directory, so the install was invisible to the task.
Measured -- the probe died in 208ms having written nothing, which is the worst
shape of failure available here, since the wrapper is what explains failures.

Pinned against `infra/main.tf` by ``tests/interfaces/cloud/test_spec.py``.
"""
NODE_PYTHON = "3.13"
NODE_PYTHON_BIN = f"/usr/local/bin/python{NODE_PYTHON}"

TASK_ID_LIMIT = 64

_UNSAFE_TASK_CHARS = re.compile(r"[^A-Za-z0-9_-]")

TASK_COMMAND_TEMPLATE = (
    "CODE_DIR=/mnt/work/code-$AZ_BATCH_TASK_ID && mkdir -p $CODE_DIR && "
    "tar xzf $AZ_BATCH_NODE_MOUNTS_DIR/shared/code/{snapshot}.tar.gz -C $CODE_DIR "
    "--no-same-owner --no-same-permissions && "
    # NOT the system python3, which is 3.10 on the pinned 22.04 image. The start
    # task installs this one and links it onto PATH; see NODE_PYTHON_BIN.
    #
    # `-u` because the wrapper's own lines are what explain a task, and a
    # buffered stdout would hold them until the process being diagnosed is gone.
    f"CODE_DIR=$CODE_DIR {NODE_PYTHON_BIN} -u $CODE_DIR/infra/run_task.py"
)


def daily_job_id(now: datetime) -> str:
    """One job per UTC day, holding that day's tasks."""
    return f"poker-{now:%Y%m%d}"


def suffixed_job_id(now: datetime) -> str:
    """The fallback id used when the day's job can no longer take tasks.

    A job that has been STOPPED answers ``JobCompleted`` to a task creation.
    Since the id is per-day, one ``just panic`` would otherwise block every
    further submission until midnight UTC.
    """
    return f"poker-{now:%Y%m%d-%H%M%S}"


def task_id(label: str, now: datetime, nonce: int) -> str:
    """Build a task id Batch will accept.

    Batch allows only alphanumerics, hyphen and underscore, and a run id or
    config name can carry neither. ``nonce`` keeps two submissions inside the
    same second apart; it is a parameter rather than a call to ``random`` so
    the id is a pure function and a test can pin it.

    Batch also REJECTS an id over 64 characters, and a rejection costs a
    snapshot upload before it is discovered. The label is what gives: the
    suffix is what keeps two submissions in one second apart, so it cannot be
    the part that gets trimmed.
    """
    safe = _UNSAFE_TASK_CHARS.sub("-", label) or "task"
    suffix = f"-{now:%H%M%S}-{nonce}"
    head = safe[: TASK_ID_LIMIT - len(suffix)].rstrip("-") or "task"
    return f"{head}{suffix}"


def task_command(snapshot: str) -> str:
    """The task command line: extract the pinned snapshot, then run the wrapper.

    The wrapper lives INSIDE the tarball, so the command line has to bootstrap
    it -- it cannot ``chmod`` a path that only exists after extraction. It
    extracts into a directory the TASK creates, because the start task runs
    elevated and tar restoring the archive root's mode onto a root-owned
    directory fails with ``Cannot change mode``. Keying on ``AZ_BATCH_TASK_ID``
    also stops two tasks on one node sharing a tree.

    The ``$``-prefixed names must survive into the node's shell unexpanded;
    only ``snapshot`` is substituted here.
    """
    return TASK_COMMAND_TEMPLATE.format(snapshot=snapshot)


@dataclass(frozen=True)
class TaskSpec:
    """One unit of work for a node, as the wrapper will read it.

    ``to`` is an ABSOLUTE iteration target, never an increment. That single
    property is what makes a Batch retry converge instead of training twice:
    ``train-static --iterations`` no-ops once the target is reached, so running
    the same task any number of times lands on the same endpoint. A relative
    target would compound on every retry.
    """

    code_snapshot: str
    op: str = TaskName.TRAIN
    config: str = ""
    to: int = 0
    run_id: str = ""
    experiment: str = ""
    arm: str = ""
    parent: str = ""
    sets: tuple[str, ...] = ()
    workers: int | None = None
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    timeout: str = DEFAULT_TIMEOUT
    eval_method: str = ""
    eval_at: str = ""
    eval_flags: tuple[str, ...] = field(default_factory=tuple)
    force_publish: bool = False
    # Stamped by `dispatch.stage_and_queue`, never by a caller: the node has no
    # `.git` (the snapshot excludes it), so the submitting machine is the only
    # witness to what code this task runs. Left empty a task still runs -- it just
    # records a null commit, which is what every cloud run did before this.
    git_commit: str = ""
    git_dirty: str = ""

    @property
    def label(self) -> str:
        """What the task id is built from: what this task DOES, not when it ran.

        The words are the KIND's -- see `src.shared.tasks`. Uniqueness is
        ``task_id``'s nonce to guarantee; this only has to be readable.
        """
        return tasks.kind(self.op).label(self)

    def environment(self) -> dict[str, str]:
        """The full RUN_* environment the node wrapper reads.

        Every key is emitted even when empty. An absent variable and an empty
        one are the same thing to the node, but emitting them all keeps the
        contract visible in one place rather than implied by which branch
        happened to set what.
        """
        return {
            "CODE_SNAPSHOT": self.code_snapshot,
            "RUN_OP": self.op,
            "RUN_CONFIG": self.config,
            "RUN_TO": str(self.to),
            "RUN_ID": self.run_id,
            "RUN_EXPERIMENT": self.experiment,
            "RUN_ARM": self.arm,
            "RUN_PARENT": self.parent,
            "RUN_SETS_JSON": json.dumps(list(self.sets)),
            "RUN_TIMEOUT": self.timeout,
            "RUN_WORKERS": "" if self.workers is None else str(self.workers),
            "RUN_CHECKPOINT_EVERY": str(self.checkpoint_every),
            "RUN_EVAL_METHOD": self.eval_method,
            "RUN_EVAL_AT": self.eval_at,
            "RUN_EVAL_FLAGS_JSON": json.dumps(list(self.eval_flags)),
            "RUN_FORCE_PUBLISH": "1" if self.force_publish else "",
            "RUN_GIT_COMMIT": self.git_commit,
            # Three-state ("1"/"0"/""), unlike the booleans above: `gitinfo`
            # distinguishes "verified clean" from "unknown", and that
            # distinction is what makes a bare hash worth recording.
            "RUN_GIT_DIRTY": self.git_dirty,
        }

    def validate(self) -> None:
        """Reject the submissions that would waste a node rather than fail fast.

        The kind-specific half lives with the kind; an unknown op is refused by
        the lookup itself, so a spec the node could not run cannot be built.
        """
        tasks.kind(self.op).validate(self)
        for override in self.sets:
            if "=" not in override:
                raise BadTaskError(f"--set expects key=value, got '{override}'.")


def utcnow() -> datetime:
    """The one clock the submission path reads, so tests can pass their own."""
    return datetime.now(UTC)
