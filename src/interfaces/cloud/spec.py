"""What a leg IS, separated from how it gets dispatched.

This module is deliberately free of the Azure SDK. Everything here is a pure
function of its arguments -- which is the point: the parts of a submission that
can be *wrong* (an iteration target that is relative instead of absolute, an
override that loses half its value, a task id that Batch rejects) are decided
here, where a test can look at them, rather than inside a string being handed
to a service.

The node contract
-----------------
``infra/run_leg.sh`` reads every value below out of the task's environment. Two
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

TRAIN = "train"
EVALUATE = "evaluate"
REPAIR_LADDER = "repair-ladder"
PRECOMPUTE = "precompute"

DEFAULT_TIMEOUT = "6h"
DEFAULT_CHECKPOINT_EVERY = 1_000_000

_UNSAFE_TASK_CHARS = re.compile(r"[^A-Za-z0-9_-]")

LEG_COMMAND_TEMPLATE = (
    "CODE_DIR=/mnt/work/code-$AZ_BATCH_TASK_ID && mkdir -p $CODE_DIR && "
    "tar xzf $AZ_BATCH_NODE_MOUNTS_DIR/shared/code/{snapshot}.tar.gz -C $CODE_DIR "
    "--no-same-owner --no-same-permissions && "
    "CODE_DIR=$CODE_DIR bash $CODE_DIR/infra/run_leg.sh"
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
    """
    safe = _UNSAFE_TASK_CHARS.sub("-", label) or "leg"
    return f"{safe}-{now:%H%M%S}-{nonce}"


def leg_command(snapshot: str) -> str:
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
    return LEG_COMMAND_TEMPLATE.format(snapshot=snapshot)


@dataclass(frozen=True)
class LegSpec:
    """One unit of work for a node, as the wrapper will read it.

    ``to`` is an ABSOLUTE iteration target, never an increment. That single
    property is what makes a Batch retry converge instead of training twice:
    ``train-static --iterations`` no-ops once the target is reached, so running
    the same leg any number of times lands on the same endpoint. A relative
    target would compound on every retry.
    """

    code_snapshot: str
    op: str = TRAIN
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

    @property
    def label(self) -> str:
        """What the task id is built from: the run being continued, else the config."""
        return self.run_id or self.config

    def environment(self) -> dict[str, str]:
        """The full RUN_* environment the node wrapper reads.

        Every key is emitted even when empty. ``run_leg.sh`` tests each with
        ``-n``, and an absent variable and an empty one are the same thing to
        it -- but emitting them all keeps the contract visible in one place
        rather than implied by which branch happened to set what.
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
        }

    def validate(self) -> None:
        """Reject the submissions that would waste a node rather than fail fast."""
        if self.op not in (TRAIN, EVALUATE, REPAIR_LADDER, PRECOMPUTE):
            raise ValueError(f"Unknown op '{self.op}'.")
        if self.op == TRAIN and not self.config and not self.run_id:
            raise ValueError(
                "A training leg needs --config (fresh run) or --run (continue an existing one)."
            )
        if self.op == TRAIN and self.to <= 0:
            raise ValueError("--to must be a positive ABSOLUTE iteration target, not an increment.")
        if self.op in (EVALUATE, REPAIR_LADDER) and not self.run_id:
            raise ValueError(f"op '{self.op}' scores an existing run, so --run is required.")
        if self.op == PRECOMPUTE and not self.config:
            raise ValueError("A precompute leg needs --config (an abstraction config stem).")
        for override in self.sets:
            if "=" not in override:
                raise ValueError(f"--set expects key=value, got '{override}'.")


def utcnow() -> datetime:
    """The one clock the submission path reads, so tests can pass their own."""
    return datetime.now(UTC)
