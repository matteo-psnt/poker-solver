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

``src/interfaces/cloud/spec.py`` writes this environment. The two files are the
same contract from opposite ends, and ``test_plan.py`` pins them together.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.shared.cloudtask import kinds
from src.shared.cloudtask.kinds import BadTaskError, TaskName

DEFAULT_TIMEOUT_SECONDS = 6 * 3600
DEFAULT_EVAL_METHOD = "exact_br"


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
    """The snapshot this task extracted, which is the EXACT answer to "what code
    is this" -- it names the bytes, uncommitted changes included, where a commit
    names a history the tree may have diverged from. It reached the node as a
    fetch instruction long before it was carried here; what it lacked was
    anywhere to be recorded."""
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
    op = env.get("RUN_OP") or TaskName.TRAIN

    plan = TaskPlan(
        op=op,
        config=env.get("RUN_CONFIG", ""),
        to=_int(env.get("RUN_TO"), 0),
        run_id=env.get("RUN_ID", ""),
        experiment=env.get("RUN_EXPERIMENT", ""),
        arm=env.get("RUN_ARM", ""),
        parent=env.get("RUN_PARENT", ""),
        sets=_json_list(env.get("RUN_SETS_JSON"), "RUN_SETS_JSON"),
        workers=_int(env.get("RUN_WORKERS"), 0) or _node_cpus(),
        checkpoint_every=_int(env.get("RUN_CHECKPOINT_EVERY"), 0),
        timeout_seconds=parse_duration(env.get("RUN_TIMEOUT")),
        eval_method=env.get("RUN_EVAL_METHOD") or DEFAULT_EVAL_METHOD,
        eval_rungs=tuple(r for r in (env.get("RUN_EVAL_AT") or "").split(",") if r),
        eval_flags=_json_list(env.get("RUN_EVAL_FLAGS_JSON"), "RUN_EVAL_FLAGS_JSON"),
        force_publish=bool(env.get("RUN_FORCE_PUBLISH")),
        git_commit=env.get("RUN_GIT_COMMIT", ""),
        git_dirty=env.get("RUN_GIT_DIRTY", ""),
        git_branch=env.get("RUN_GIT_BRANCH", ""),
        code_snapshot=env.get("CODE_SNAPSHOT", ""),
    )
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


def _int(raw: str | None, default: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _json_list(raw: str | None, name: str) -> tuple[str, ...]:
    """A JSON array, not a space-separated string.

    The elements are LISTS whose members may contain ``=``, a space or a
    newline -- a config override's value routinely does. The old
    ``for kv in $RUN_SETS`` split on whitespace and silently cut such a value in
    half; the shell port of that fix needed a NUL-separated temp file and
    ``read -d ''`` to get back to what one ``json.loads`` does here.

    A malformed payload is FATAL, never an empty list. See
    :class:`BadEnvironment`.
    """
    if not raw:
        return ()
    try:
        decoded = json.loads(raw)
    except ValueError as error:
        raise BadEnvironmentError(f"could not decode {name}: {error}") from error
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise BadEnvironmentError(f"{name} must be a JSON array of strings, got: {raw!r}")
    return tuple(item for item in decoded if item)


_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(raw: str | None) -> int:
    """``6h`` / ``90m`` / ``3600`` -> seconds.

    The wall-clock ceiling for the training process itself. The task-level
    ``maxWallClockTime`` (P1D) is not a backstop for a hang -- it is longer than
    any task is meant to run, so a wedged process bills a full node-day before
    Batch acts. One task proved it: training died, the process could not exit,
    and the task stayed ``running`` indefinitely.
    """
    text = (raw or "").strip().lower()
    if not text:
        return DEFAULT_TIMEOUT_SECONDS
    unit = _UNITS.get(text[-1])
    seconds = _int(text, 0) if unit is None else _int(text[:-1], 0) * unit
    # A non-positive ceiling would make the guard fire before the trainer has
    # started, which reads as a hang (124) rather than as the bad input it is.
    return seconds if seconds > 0 else DEFAULT_TIMEOUT_SECONDS
