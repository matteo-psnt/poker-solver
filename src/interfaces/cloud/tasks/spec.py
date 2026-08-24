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
see :mod:`src.shared.cloudtask.node.plan`, which is the same contract from the node's
end and is pinned against this one by ``tests/shared/cloudtask/node/test_plan.py``. Two
of them are JSON-encoded rather than passed raw, and the reason is not the one
the old shell had. The ``az`` CLI needed hex because
``--environment-settings`` parses ``KEY=VALUE`` and a config override's value
contains ``=``; the SDK has no such problem. JSON survives here because
overrides and eval flags are *lists whose elements may contain spaces*, and the
old space-joined form silently split them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.shared.cloudtask import kinds, wire
from src.shared.cloudtask.kinds import BadTaskError, TaskName

DEFAULT_TIMEOUT = "6h"
# Measured 08-22 at 16 workers: a 1M chunk is ~18 s, of which ~2 s is the
# checkpoint write and ~1-2 s the worker respawn. 5M keeps that under 5%
# and bounds a killed run's loss at ~90 s of work -- the same bound in TIME
# that 1M gave at the old rate.
DEFAULT_CHECKPOINT_EVERY = 5_000_000

# NOT the OS python3 (3.10 on the pinned image). The start task installs this one
# at a fixed absolute path, so no uv is needed at task time -- `uv run` here
# resolved its interpreter under HOME, which differs between the elevated start
# task and the task's own auto-user. Pinned against `infra/main.tf` by
# `tests/interfaces/cloud/test_spec.py`.
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


def daily_job_id(now: datetime, pool_suffix: str = "") -> str:
    """One job per UTC day AND POOL: a Batch job is bound to one pool at
    creation, so the big pool's tasks need their own."""
    return f"poker-{now:%Y%m%d}{pool_suffix}"


def suffixed_job_id(now: datetime, pool_suffix: str = "") -> str:
    """The fallback id used when the day's job can no longer take tasks.

    A job that has been STOPPED answers ``JobCompleted`` to a task creation.
    Since the id is per-day, one ``just panic`` would otherwise block every
    further submission until midnight UTC.
    """
    return f"poker-{now:%Y%m%d-%H%M%S}{pool_suffix}"


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
    universe_boards: int = 0
    universe_seed: int = 0
    dtype: str = ""
    warm_start_from: str = ""
    warm_start_weight: int = 0
    warm_start_at: int = 0
    warm_start_shape: str = ""
    equity_prior_weight: int = 0
    equity_prior_temperature: float = 0.0
    force_publish: bool = False
    # Stamped by `dispatch.stage_and_queue`, never by a caller: the node has no
    # `.git` (the snapshot excludes it), so the submitting machine is the only
    # witness to what code this task runs. Left empty a task still runs -- it just
    # records a null commit, which is what every cloud run did before this.
    git_commit: str = ""
    git_dirty: str = ""
    # The branch, because a commit does not identify an EXPERIMENT here. Work
    # runs in several worktrees at once and each carries its change uncommitted
    # while it is being iterated on, so two arms are routinely the same hash
    # with the same dirty bit. See `shared.gitinfo.BRANCH_ENV`.
    git_branch: str = ""

    @property
    def label(self) -> str:
        """What the task id is built from: what this task DOES, not when it ran.

        The words are the KIND's -- see `src.shared.cloudtask.kinds`. Uniqueness is
        ``task_id``'s nonce to guarantee; this only has to be readable.
        """
        return kinds.kind(self.op).label(self)

    def environment(self) -> dict[str, str]:
        """The full RUN_* environment the node wrapper reads.

        Derived from :data:`src.shared.cloudtask.wire.KEYS`, which the node
        decodes through as well, so this end cannot emit a key the other does
        not read or spell one of them differently.
        """
        return wire.encode(self)

    def validate(self) -> None:
        """Reject the submissions that would waste a node rather than fail fast.

        The kind-specific half lives with the kind; an unknown op is refused by
        the lookup itself, so a spec the node could not run cannot be built.
        """
        kinds.kind(self.op).validate(self)
        for override in self.sets:
            if "=" not in override:
                raise BadTaskError(f"--set expects key=value, got '{override}'.")
        # Both shape a guess that only exists when a weight asks for one, and
        # the argv builder skips them without it. Silently dropping them would
        # train a plain control under the variant's arm label -- the failure
        # that has twice cost a whole sweep, so it raises at submit instead.
        if self.equity_prior_temperature and not self.equity_prior_weight:
            raise BadTaskError(
                "--equity-prior-temperature shapes the equity prior, and without "
                "--equity-prior there is no prior to shape. This would train a "
                "control under a variant's arm name."
            )


def utcnow() -> datetime:
    """The one clock the submission path reads, so tests can pass their own."""
    return datetime.now(UTC)
