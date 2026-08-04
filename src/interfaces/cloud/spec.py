"""What a leg IS, separated from how it gets dispatched.

This module is deliberately free of the Azure SDK. Everything here is a pure
function of its arguments -- which is the point: the parts of a submission that
can be *wrong* (an iteration target that is relative instead of absolute, an
override that loses half its value, a task id that Batch rejects) are decided
here, where a test can look at them, rather than inside a string being handed
to a service.

The node contract
-----------------
``infra/run_leg.py`` reads every value below out of the task's environment --
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

from src.shared.describe import compact_count, flag_value

TRAIN = "train"
EVALUATE = "evaluate"
REPAIR_LADDER = "repair-ladder"
PRECOMPUTE = "precompute"

DEFAULT_TIMEOUT = "6h"
DEFAULT_CHECKPOINT_EVERY = 1_000_000

"""Task ids
--------
``TASK_ID_LIMIT`` is Batch's, not ours. ``_OP_WORDS`` shortens the op for the
one place length is scarce; the op itself stays the long form everywhere else.
"""
TASK_ID_LIMIT = 64
_OP_WORDS = {TRAIN: "train", EVALUATE: "score", REPAIR_LADDER: "repair", PRECOMPUTE: "precompute"}

_UNSAFE_TASK_CHARS = re.compile(r"[^A-Za-z0-9_-]")

LEG_COMMAND_TEMPLATE = (
    "CODE_DIR=/mnt/work/code-$AZ_BATCH_TASK_ID && mkdir -p $CODE_DIR && "
    "tar xzf $AZ_BATCH_NODE_MOUNTS_DIR/shared/code/{snapshot}.tar.gz -C $CODE_DIR "
    "--no-same-owner --no-same-permissions && "
    # The node's SYSTEM python3 -- 3.10 on the pinned image, and the only
    # interpreter that exists before `uv sync`. `-u` because the wrapper's own
    # lines are the ones that explain a leg, and a buffered stdout would hold
    # them until the process that is being diagnosed has already ended.
    "CODE_DIR=$CODE_DIR python3 -u $CODE_DIR/infra/run_leg.py"
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
    safe = _UNSAFE_TASK_CHARS.sub("-", label) or "leg"
    suffix = f"-{now:%H%M%S}-{nonce}"
    head = safe[: TASK_ID_LIMIT - len(suffix)].rstrip("-") or "leg"
    return f"{head}{suffix}"


def run_token(run_id: str) -> str:
    """A short but still recognisable form of a run id, for use inside a task id.

    Run ids are themselves built from task ids (``plan.train_run_id``), so
    pasting one in whole would grow every id by the length of the last one.
    The first and last segments carry the config and the discriminator --
    ``run-production-025433-1095`` -> ``production-1095``. The timestamp in the
    middle is the part nobody reads and the part that makes ids look alike.
    """
    stem = run_id.removeprefix("run-")
    parts = [part for part in stem.split("-") if part]
    return f"{parts[0]}-{parts[-1]}" if len(parts) > 2 else stem


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
    # Stamped by `dispatch.stage_and_queue`, never by a caller: the node has no
    # `.git` (the snapshot excludes it), so the submitting machine is the only
    # witness to what code this leg runs. Left empty a leg still runs -- it just
    # records a null commit, which is what every cloud run did before this.
    git_commit: str = ""
    git_dirty: str = ""

    @property
    def label(self) -> str:
        """What the task id is built from: what this leg DOES, not when it ran.

        Every discriminating field is already on this spec, and returning the
        bare run id threw all of them away. Three evaluations of ONE checkpoint
        at three board seeds became three ids differing only in a timestamp and
        a nonce, so nothing downstream -- the console, `legs`, the log viewer --
        could say which was which; the mapping lived only in the head of
        whoever submitted them.

        Uniqueness is ``task_id``'s nonce to guarantee. This only has to be
        readable, which is why it carries the knob that usually differs rather
        than every knob that could.
        """
        words = [
            _OP_WORDS.get(self.op, self.op),
            run_token(self.run_id) if self.run_id else self.config,
        ]
        if self.op == TRAIN and self.to:
            words.append(f"to{compact_count(self.to)}")
        if self.op == EVALUATE:
            if self.eval_at.isdigit():
                words.append(compact_count(int(self.eval_at)))
            seed = flag_value(self.eval_flags, "--br-board-seed")
            if seed:
                words.append(f"seed{seed}")
        if self.arm:
            words.append(self.arm)
        return "-".join(word for word in words if word)

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
        """Reject the submissions that would waste a node rather than fail fast."""
        if self.op not in (TRAIN, EVALUATE, REPAIR_LADDER, PRECOMPUTE):
            raise ValueError(f"Unknown op '{self.op}'.")
        if self.op == TRAIN and not self.config:
            # A CONTINUING leg needs it too. The config builds the tree and the
            # solver and the checkpoint stores neither, so `--run x` without a
            # config reached the node and died on
            # `Config file not found: config/training/.yaml` -- after a snapshot
            # upload, a ~3-minute pool spin-up and every Batch retry. This used
            # to accept it, and the justfile documented it as the way to
            # continue a run.
            raise ValueError(
                "A training leg needs --config, a CONTINUING one included: the config "
                "builds the tree and the solver, and the checkpoint stores neither."
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
