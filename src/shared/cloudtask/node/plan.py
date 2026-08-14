"""The task's environment, turned into the command line it will run.

Pure: no filesystem, no subprocess, no clock. That is what lets a test look at
the exact argv a task would execute, which used to be knowable only by reading
shell and hoping. The defect that motivated the check is small and expensive --
a shared array carrying ``--workers`` splatted into a command that declares no
such flag, three Batch retries each dead about four seconds in, after a code
snapshot upload and a ~3-minute pool spin-up.

Deliberately does NOT import the command layer. ``.importlinter`` forbids
``src.shared -> src.interfaces``, and the node has no dependencies installed
when this runs anyway. The argv is emitted as data; the *test* imports
``COMMANDS`` and checks every flag against the real parser.

``src/interfaces/cloud/tasks/spec.py`` writes this environment, and neither end
spells the keys any more: both derive from :data:`src.shared.cloudtask.wire.KEYS`,
which is the one declaration of what crosses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.shared.cloudtask import kinds, wire
from src.shared.cloudtask.kinds import BadTaskError, TaskName
from src.shared.cloudtask.wire import (
    DEFAULT_EVAL_METHOD,
    DEFAULT_TIMEOUT_SECONDS,
    parse_duration,
)

__all__ = ["DEFAULT_EVAL_METHOD", "DEFAULT_TIMEOUT_SECONDS", "parse_duration"]


class BadEnvironmentError(Exception):
    """The task's environment cannot be turned into a runnable task.

    Fatal on purpose, and never a silent default. A malformed override payload
    that decoded to zero overrides would train an experiment arm with the BASE
    config -- an arm silently running as its own control, and recorded that way
    in ``.run.json``. That class of silent rebucketing has already cost one
    curve.
    """


@dataclass(frozen=True)
class TaskPlan:
    """One task, as the node will execute it."""

    op: str
    config: str = ""
    to: int = 0
    run_id: str = ""
    experiment: str = ""
    arm: str = ""
    parent: str = ""
    sets: tuple[str, ...] = ()
    workers: int = 1
    checkpoint_every: int = 0
    universe_boards: int = 0
    universe_seed: int = 0
    dtype: str = ""
    warm_start_from: str = ""
    warm_start_weight: int = 0
    warm_start_at: int = 0
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    eval_method: str = DEFAULT_EVAL_METHOD
    eval_rungs: tuple[str, ...] = ()
    eval_flags: tuple[str, ...] = field(default_factory=tuple)
    force_publish: bool = False
    # Not used to build an argv -- `gitinfo` reads these straight out of the
    # environment, which the `uv run` child inherits, so nothing has to thread
    # them through a command line. Carried here so the wrapper can SAY what code
    # it is running: a task log that names its commit is the only place the
    # answer appears while the task is alive.
    git_commit: str = ""
    git_dirty: str = ""
    git_branch: str = ""
    """The snapshot this task extracted -- the exact bytes, uncommitted changes
    included, where a commit names a history the tree may have diverged from."""
    code_snapshot: str = ""
    """Where a build writes its own progress, filled in by the wrapper because
    only the node knows its scratch directory."""
    progress_path: str = ""

    @property
    def train_run_id(self) -> str:
        """The run a training task writes to.

        Derived from the task when none was given, so a Batch RETRY -- which
        keeps the same task id -- continues this run rather than starting a
        second one from zero. That is what makes a retry safe here.
        """
        return self.run_id or f"run-{os.environ.get('AZ_BATCH_TASK_ID', 'local')}"

    @property
    def commands(self) -> list[list[str]]:
        """The argv(s) this task runs, from its KIND -- so nothing here branches.

        A list: scoring a ladder is one command per rung, and the wrapper
        accounts for each separately.
        """
        return kinds.kind(self.op).commands(self)

    @property
    def provenance(self) -> str:
        """What code this task runs, for the log.

        The node has no `.git` -- the code snapshot excludes it -- so this is
        the submitter's answer, passed down. Until it existed, every
        cloud-trained run and every cloud-run evaluation recorded a null
        commit.

        The branch and the snapshot are named alongside it because the commit
        alone is not an identity here: several worktrees investigate different
        things off the same hash, each carrying its change uncommitted, so
        "c13dcb7 (DIRTY tree)" has described four different programs. The
        snapshot is the exact one.
        """
        if not self.git_commit:
            described = "unknown (nothing stamped this task)"
        else:
            state = {"1": " (DIRTY tree)", "0": " (clean tree)"}.get(
                self.git_dirty, " (tree state unknown)"
            )
            branch = f" on {self.git_branch}" if self.git_branch else ""
            described = self.git_commit[:12] + branch + state
        # Last, and unconditional: it is the only part that stays exact when the
        # tree was dirty, which is the normal state of an experiment worktree.
        return f"{described}, snapshot {self.code_snapshot or '?'}"


def parse_environment(environ: dict[str, str] | None = None) -> TaskPlan:
    """Read the ``RUN_*`` contract out of the task's environment.

    An absent variable and an empty one mean the same thing, which is why
    ``spec.TaskSpec.environment`` emits every key: the contract stays visible in
    one place rather than implied by which branch happened to set what.
    """
    env = dict(os.environ if environ is None else environ)
    try:
        fields = wire.decode(env)
    except wire.BadWireValueError as error:
        raise BadEnvironmentError(str(error)) from error

    # The two things only this end can supply, kept visible rather than buried
    # in a codec: an environment with no `RUN_OP` has always meant training, and
    # the node is the only place that knows its own core count.
    fields["op"] = fields["op"] or TaskName.TRAIN
    fields["workers"] = fields["workers"] or _node_cpus()

    plan = TaskPlan(**fields)
    _validate(plan)
    return plan


def _validate(plan: TaskPlan) -> None:
    """Refuse here rather than let argparse exit 2 several minutes in.

    The same check the submitter ran, from the same kind -- so the node cannot
    accept something the submit path would have refused, and neither can drift
    from the other. Re-raised as a `BadEnvironmentError` because that is what
    the wrapper's exit accounting knows how to report.
    """
    try:
        kinds.kind(plan.op).validate(plan)
    except BadTaskError as error:
        raise BadEnvironmentError(str(error)) from error


def _node_cpus() -> int:
    """The node is the only place that knows its own core count.

    Filled in HERE rather than left to the CLI's local-friendly default of 1,
    and never allowed to fail: a missing core count must degrade, not kill the
    task.
    """
    return os.cpu_count() or 1
