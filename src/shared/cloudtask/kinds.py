"""What KIND of work a task is, and everything that differs because of it.

``op`` was a string, and the behaviour keyed off it lived in four modules: the
argv in ``node.plan``, the label and the submit-time validation in
``cloud.spec``, the retry count in ``cloud.dispatch``, the one-line description
in ``task_log``. Adding a kind meant finding all four, and nothing failed if you
found three -- the missing branch was a task that ran with the wrong argv, or a
retry that billed three full runs to fail three times.

A kind is one class here instead, and :data:`KINDS` at the bottom of this module
is the list of them. ``tests/shared/cloudtask/test_kinds.py`` fails if that list
and :class:`TaskName` disagree.

Nothing here touches the filesystem or a service. ``sample`` is handed the state
the node has already gathered rather than reading it, which is what keeps this
importable from both ends of the wire -- the submitter builds a spec with it,
the node builds an argv with it -- and testable without either.

Stdlib only: ``node.plan`` imports this and runs before ``uv sync``.
"""

from __future__ import annotations

import abc
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Protocol, cast


class TaskName(StrEnum):
    """The wire vocabulary, closed on purpose.

    ``StrEnum`` because this IS the string: it travels as ``RUN_OP`` in a task's
    environment and as ``op`` in every record, and a member compares equal to
    its own text, so nothing converts at either boundary. It also validates on
    parse, which is exactly what the node's environment read wants.

    Closed at the SUBMIT end -- an op the node cannot run must not reach a node.
    Readers are deliberately open: the task log still holds ``vector-sweep`` and
    ``train-vector`` from work that no longer exists, and listing history must
    not raise on its own past. Readers use :func:`kind_of`, which returns
    ``None``, not ``TaskName(...)``, which raises.
    """

    TRAIN = "train"
    TRAIN_VECTOR = "train-vector"
    EVALUATE = "evaluate"
    PRECOMPUTE = "precompute"
    VECTOR_SWEEP = "vector-sweep"


class BadTaskError(ValueError):
    """A task that would waste a node, rejected before it costs one.

    A ``ValueError`` because that is what it is, and because the submit path
    already treats one as a refusal rather than a crash.
    """


class TaskFields(Protocol):
    """What a kind's :meth:`TaskKind.validate` reads, from EITHER end of the wire.

    Structural on purpose. The submit end passes
    ``interfaces.cloud.spec.TaskSpec`` and the node end passes
    ``shared.node.plan.TaskPlan``, and ``.importlinter`` forbids
    ``src.shared -> src.interfaces`` -- so this module cannot name either class,
    not even under ``TYPE_CHECKING``. A Protocol is how the seam gets a type
    without the import that would invert the layering.

    The INTERSECTION of the two shapes, deliberately. Both ends validate through
    the same kind -- that is what stops the node accepting what the submitter
    would have refused -- so a field only one of them carries cannot be part of
    the check.

    Read-only properties rather than plain attributes, because both
    implementations are frozen dataclasses and a mutable protocol member would
    refuse them.
    """

    @property
    def config(self) -> str: ...
    @property
    def to(self) -> int: ...
    @property
    def run_id(self) -> str: ...


class Submission(TaskFields, Protocol):
    """A task as the SUBMITTER holds it: what a label is built from.

    ``eval_at`` is a single rung here and ``eval_rungs`` a tuple on the node,
    which is the one place the two shapes genuinely differ rather than merely
    overlap -- one task scores the ladder the submitter named.
    """

    @property
    def arm(self) -> str: ...
    @property
    def eval_at(self) -> str: ...
    @property
    def eval_flags(self) -> Sequence[str]: ...
    # Read by TrainVectorTask.validate: the sampled boards are the chance layer,
    # so a submission that omits them names no game and must be refused here.
    @property
    def universe_boards(self) -> int: ...


class NodePlan(TaskFields, Protocol):
    """A task as the NODE holds it: what an argv and a progress sample are built from.

    Wider than :class:`Submission` because this end has resolved what the
    submitter left open -- the worker count the node actually has, the run id a
    retry must continue, the scratch path a build reports progress into.
    """

    @property
    def train_run_id(self) -> str: ...
    @property
    def workers(self) -> int: ...
    @property
    def checkpoint_every(self) -> int: ...
    @property
    def experiment(self) -> str: ...
    @property
    def arm(self) -> str: ...
    @property
    def parent(self) -> str: ...
    @property
    def sets(self) -> Sequence[str]: ...
    @property
    def eval_method(self) -> str: ...
    @property
    def eval_rungs(self) -> Sequence[str]: ...
    @property
    def eval_flags(self) -> Sequence[str]: ...
    @property
    def progress_path(self) -> str: ...
    @property
    def universe_boards(self) -> int: ...
    @property
    def universe_seed(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def warm_start_from(self) -> str: ...
    @property
    def warm_start_weight(self) -> int: ...
    @property
    def warm_start_at(self) -> int: ...


@dataclass(frozen=True)
class Progress:
    """How far along a task is, in whatever it counts.

    Deliberately not a bare percentage. A fraction cannot say WHAT is being
    counted, and the unit is what makes "165 visits per infoset" legible as
    nowhere near the 1e3 CFR wants, rather than as a number beside a bar.
    """

    done: float
    total: float
    unit: str

    @property
    def fraction(self) -> float:
        """Clamped: a resumed task can pass its own target, and a bar drawn past
        its end reads as a rendering bug rather than as a no-op."""
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.done / self.total))

    @property
    def phrase(self) -> str:
        return f"{compact(self.done)} / {compact(self.total)} {self.unit}"

    def as_record(self) -> dict[str, Any]:
        return {"done": self.done, "total": self.total, "unit": self.unit}

    @staticmethod
    def from_record(raw: object) -> Progress | None:
        """Tolerant: read back from a share record an older wrapper may not have
        written, or may have been killed halfway through writing."""
        if not isinstance(raw, Mapping):
            return None
        # All the narrowing on offer: the record is untyped JSON off an SMB
        # share, so the KEYS are checked by the try below, not by the checker.
        record = cast("Mapping[str, Any]", raw)
        try:
            return Progress(
                float(record["done"]), float(record["total"]), str(record.get("unit") or "")
            )
        except (KeyError, TypeError, ValueError):
            return None


def compact(value: float) -> str:
    """``150000000`` -> ``150M``. These counts are unreadable in full digits."""
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


@dataclass(frozen=True)
class Sample:
    """What one finished task achieved, for predicting the next one.

    A RATE, not a duration. A task training to 200M is not predicted by the
    median duration of tasks that trained to 5M -- but their rate predicts it
    fine, which is the whole reason this carries units as well as seconds.

    ``workers`` because throughput scales with them, so a history that mixes
    counts predicts nothing. Samples are MATCHED on it rather than divided by
    it: scaling is sublinear and saturates -- 16 workers and 32 measured within
    noise of each other past 10M iterations -- so normalising by dividing would
    be confidently wrong in the range that matters.
    """

    units: float
    seconds: float
    workers: int = 0

    @property
    def rate(self) -> float:
        return self.units / self.seconds if self.seconds > 0 else 0.0


def comparable(history: Sequence[Sample], workers: int) -> list[Sample]:
    """Past tasks worth predicting from: the same worker count if any ran at it.

    Falling back to the whole history rather than refusing is deliberate. A
    rough estimate beats none, and the first task at a new worker count would
    otherwise never get one.
    """
    usable = [sample for sample in history if sample.seconds > 0]
    matched = [sample for sample in usable if sample.workers == workers]
    return matched or usable


class TaskKind(abc.ABC):
    """One kind of work, and every way it differs from the others."""

    name: ClassVar[TaskName]
    unit: ClassVar[str]
    """A file this kind writes its progress into, relative to the node's scratch.
    Empty when the work is observable another way; the wrapper branches on it."""
    progress_file: ClassVar[str] = ""
    """Batch retries. Work cheap to repeat wants them -- training resumes from
    its last published rung, scoring is idempotent. Work with no
    partial-progress marker does not."""
    retries: ClassVar[int] = 3

    @abc.abstractmethod
    def validate(self, task: TaskFields) -> None:
        """Reject a submission that cannot run, before it reaches a node."""

    @abc.abstractmethod
    def commands(self, plan: NodePlan) -> list[list[str]]:
        """The `poker-solver` argv(s) this task runs, in order.

        A list because scoring a ladder is one command per rung; the node runs
        them in sequence and accounts for each.
        """

    @abc.abstractmethod
    def label(self, task: Submission) -> str:
        """The words a task id is built from -- what this task DOES."""

    @abc.abstractmethod
    def describe(self, record: Mapping[str, Any]) -> str:
        """One phrase saying what this task did, from its recorded fields."""

    @abc.abstractmethod
    def sample(self, plan: NodePlan, state: Mapping[str, object]) -> Progress | None:
        """How far along, from state the node has already gathered.

        ``None`` when the kind cannot yet say -- honest, and renders as no bar
        rather than as a bar stuck at zero.
        """

    def estimate(
        self,
        progress: Progress | None,
        elapsed: float,
        history: Sequence[Sample] = (),
        workers: int = 0,
    ) -> float | None:
        """Seconds remaining, or ``None`` when there is nothing to go on.

        Four sources, best first:

        1. The task's OWN rate. Once it has reported progress it is measuring
           itself on the machine it is actually running on, at the worker count
           it actually got -- which beats any prior, and needs no history.
        2. The rate of comparable past tasks, against the work remaining. This
           is why a sample carries units: it transfers across tasks of different
           SIZE, where a median duration does not.
        3. Their median duration, for a kind that cannot report progress at all
           and so has no "remaining" to scale.
        4. Nothing, said as ``None`` rather than guessed.

        The 2% floor on (1) is not arbitrary: extrapolating from the first
        fraction of a percent turns startup cost into a wildly wrong total.
        """
        if progress is not None and progress.fraction > 0.02 and elapsed > 0:
            return elapsed * (1.0 - progress.fraction) / progress.fraction

        usable = comparable(history, workers)
        rates = [sample.rate for sample in usable if sample.rate > 0]
        if progress is not None and rates:
            remaining = max(0.0, progress.total - progress.done)
            return remaining / statistics.median(rates)
        if usable:
            return max(0.0, statistics.median([sample.seconds for sample in usable]) - elapsed)
        return None


class TrainTask(TaskKind):
    """Train a run to an ABSOLUTE iteration target."""

    name = TaskName.TRAIN
    unit = "iterations"

    def validate(self, task: TaskFields) -> None:
        if not task.config:
            # A CONTINUING task needs it too: the config builds the tree and the
            # solver, and the checkpoint stores neither. `--run x` without one
            # reached the node and died on `Config file not found` -- after a
            # snapshot upload, a pool spin-up and every retry.
            raise BadTaskError("a training task needs a config, even when continuing a run")
        if task.to <= 0:
            raise BadTaskError("the iteration target is ABSOLUTE and must be positive")

    def commands(self, plan: NodePlan) -> list[list[str]]:
        argv = [
            "train-static",
            "--config",
            plan.config,
            "--iterations",
            str(plan.to),
            "--run",
            plan.train_run_id,
            # Never omitted. "Empty means all CPUs" was documented but never
            # implemented: `train-static` defaults --workers to 1, so an omitted
            # count trained SINGLE-THREADED on a 16-vCPU node -- a ~16x loss that
            # reads as a slow task rather than a misconfiguration.
            "--workers",
            str(plan.workers),
        ]
        if plan.checkpoint_every:
            argv += ["--checkpoint-every", str(plan.checkpoint_every)]
        # Seeding is a property of a FRESH run; train_static ignores it when
        # continuing, so a retry cannot lay the prior back over real progress.
        if plan.warm_start_from:
            argv += ["--warm-start-from", plan.warm_start_from]
            if plan.warm_start_weight:
                argv += ["--warm-start-weight", str(plan.warm_start_weight)]
            # The rung is part of the prior's identity: board-free quality is not
            # monotone, so seeding from the manifest's current rung silently uses
            # a different strategy than the one that was measured.
            if plan.warm_start_at:
                argv += ["--warm-start-at", str(plan.warm_start_at)]
        # Appended only when set: `--arm ""` records an arm literally named
        # empty string rather than an unaffiliated run.
        for flag, value in (
            ("--experiment", plan.experiment),
            ("--arm", plan.arm),
            ("--parent", plan.parent),
        ):
            if value:
                argv += [flag, value]
        for override in plan.sets:
            argv += ["--set", override]
        return [argv]

    def label(self, task: Submission) -> str:
        words = ["train", _subject(task)]
        if task.to:
            words.append(f"to{compact(task.to)}")
        if task.arm:
            words.append(task.arm)
        return "-".join(word for word in words if word)

    def describe(self, record: Mapping[str, Any]) -> str:
        target = str(record.get("target_iteration") or "")
        return f"train ->{compact(int(target))}" if target.isdigit() and target != "0" else "train"

    def sample(self, plan: NodePlan, state: Mapping[str, object]) -> Progress | None:
        """Iterations done against the target, from the checkpoint manifest.

        The one kind that can already answer, because training writes a
        checkpoint as it goes and the manifest names its iteration.
        """
        done = state.get("iteration")
        if not isinstance(done, int | float) or plan.to <= 0:
            return None
        return Progress(float(done), float(plan.to), self.unit)


class TrainVectorTask(TaskKind):
    """Train a board-free blueprint over the whole tree at once.

    The same absolute-target contract as :class:`TrainTask`, and two differences
    that are not cosmetic.

    It takes NO ``--workers``. The kernel is one process, and splatting a shared
    array carrying that flag into a command declaring none is the defect this
    module exists to make impossible -- three retries, each dead four seconds in.

    The universe knobs have no scalar analogue. The sampled boards estimate the
    bucket-transition matrices, so they are the chance layer, and therefore the
    GAME rather than the schedule. An unspecified universe would silently pick
    which game the task solves, so it is refused instead of defaulted.
    """

    name = TaskName.TRAIN_VECTOR
    unit = "iterations"

    def validate(self, task: Any) -> None:
        if not task.config:
            raise BadTaskError("a training task needs a config, even when continuing a run")
        if task.to <= 0:
            raise BadTaskError("the iteration target is ABSOLUTE and must be positive")
        if task.universe_boards <= 0:
            raise BadTaskError(
                "a board-free task needs --universe-boards: the sampled boards define "
                "the chance layer, and so the game it solves"
            )

    def commands(self, plan: Any) -> list[list[str]]:
        argv = [
            "train-vector",
            "--config",
            plan.config,
            "--iterations",
            str(plan.to),
            "--run",
            plan.train_run_id,
        ]
        for flag, value in (
            ("--universe-boards", plan.universe_boards),
            ("--universe-seed", plan.universe_seed),
            ("--checkpoint-every", plan.checkpoint_every),
        ):
            if value:
                argv += [flag, str(value)]
        if plan.dtype:
            argv += ["--dtype", plan.dtype]
        for flag, value in (
            ("--experiment", plan.experiment),
            ("--arm", plan.arm),
            ("--parent", plan.parent),
        ):
            if value:
                argv += [flag, value]
        for override in plan.sets:
            argv += ["--set", override]
        return [argv]

    def label(self, task: Any) -> str:
        words = ["vector", _subject(task)]
        if task.to:
            words.append(f"to{compact(task.to)}")
        if task.arm:
            words.append(task.arm)
        return "-".join(word for word in words if word)

    def describe(self, record: Mapping[str, Any]) -> str:
        target = str(record.get("target_iteration") or "")
        if target.isdigit() and target != "0":
            return f"board-free ->{compact(int(target))}"
        return "board-free"

    def sample(self, plan: Any, state: Mapping[str, Any]) -> Progress | None:
        """Same manifest as the scalar trainer: it writes ordinary checkpoints."""
        done = state.get("iteration")
        if not isinstance(done, int | float) or plan.to <= 0:
            return None
        return Progress(float(done), float(plan.to), self.unit)


class EvaluateTask(TaskKind):
    """Score published rungs of an existing run."""

    name = TaskName.EVALUATE
    unit = "board branches"
    progress_file = "evaluate-progress.json"

    def validate(self, task: TaskFields) -> None:
        if not task.run_id:
            raise BadTaskError("an evaluation works on an existing run, so it needs a run id")

    def commands(self, plan: NodePlan) -> list[list[str]]:
        """One command per rung. ``eval_flags`` is the submitter's passthrough
        (``score --run r -- --br-flops 8``), validated where it is built because
        its contents are unknowable here."""
        commands = []
        for rung in list(plan.eval_rungs) or [""]:
            argv = ["evaluate", "--run", plan.run_id, "--method", plan.eval_method]
            if rung:
                argv += ["--at", rung]
            # THE NODE'S CORES. `--workers` defaults to 1 and nothing here passed
            # it, so every evaluation ever run on the pool used ONE core of a
            # 16-core box: `exact_br` took its serial path and `lbr` its
            # single-process one. Ahead of `eval_flags`, so an explicit
            # `--workers` in the passthrough still wins -- argparse takes the last.
            if plan.workers > 0:
                argv += ["--workers", str(plan.workers)]
            if work := plan.progress_path:
                argv += ["--progress-file", work]
            commands.append(argv + list(plan.eval_flags))
        return commands

    def label(self, task: Submission) -> str:
        words = ["score", _subject(task)]
        rung = str(task.eval_at or "")
        if rung.isdigit():
            words.append(compact(int(rung)))
        seed = _flag(task.eval_flags, "--br-board-seed")
        if seed:
            words.append(f"seed{seed}")
        return "-".join(word for word in words if word)

    def describe(self, record: Mapping[str, Any]) -> str:
        rung = str(record.get("eval_at") or "")
        if not rung.isdigit():
            return "evaluate"
        seed = _flag(record.get("eval_flags") or (), "--br-board-seed")
        detail = f"evaluate @{compact(int(rung))}"
        return f"{detail} seed{seed}" if seed else detail

    def sample(self, plan: NodePlan, state: Mapping[str, object]) -> Progress | None:
        """Top-level flop branches walked, against the branches this score needs.

        RUNGS WERE THE WRONG UNIT. `score` submits one task per rung, so the
        denominator was 1 and the bar read 0% from the first second to the last
        of a ~10-minute score -- a long evaluation and a hung one looked
        identical, which is the single thing a bar exists to distinguish.

        Flop branches are the outermost thing the walk counts: four walks
        (responder seat x button) of `--br-flops` each, so the default 8 gives 32
        where there was 1, and `--br-flops 64` gives 256. Deeper than that is
        genuinely off limits -- the turn and river deals are the walk's inner
        loops and a counter there WOULD be in the hot path.

        Two state shapes reach here and both are wanted. The watcher publishes
        `{done, total}` from the file the evaluator writes, which is the bar
        while it runs; the handler publishes `{scored}` after each rung, which
        closes it at the end and is all there is before the first branch lands.
        """
        walked, branches = state.get("done"), state.get("total")
        if isinstance(walked, int | float) and isinstance(branches, int | float) and branches > 0:
            return Progress(float(walked), float(branches), self.unit)
        # Nothing walked yet, or a method that does not report branches at all:
        # fall back to the rung count, which is what this reported before.
        rungs = len(plan.eval_rungs)
        if rungs <= 0:
            return None
        scored = state.get("scored")
        done = float(scored if isinstance(scored, int) else 0)
        return Progress(done, float(rungs), "rungs")


class PrecomputeTask(TaskKind):
    """Build a card abstraction."""

    name = TaskName.PRECOMPUTE
    unit = "streets"
    """NOT retried. A precompute has no partial-progress marker -- `metadata.json`
    is written only on success -- so a retry restarts the whole enumeration, and
    a deterministic failure would bill three full runs to fail three times."""
    retries = 0

    def validate(self, task: TaskFields) -> None:
        if not task.config:
            raise BadTaskError("a precompute task needs an abstraction config")

    progress_file = "precompute-progress.json"

    def commands(self, plan: NodePlan) -> list[list[str]]:
        argv = ["precompute", "--config", plan.config, "--json"]
        # Where the build reports street completion, and where `sample` reads it
        # back. Node-local: it is a heartbeat, not a record.
        work = plan.progress_path
        return [[*argv, "--progress-file", work] if work else argv]

    def label(self, task: Submission) -> str:
        return f"precompute-{task.config}"

    def describe(self, record: Mapping[str, Any]) -> str:
        return f"precompute {record.get('config') or ''}".strip()

    def sample(self, plan: NodePlan, state: Mapping[str, object]) -> Progress | None:  # noqa: ARG002
        """Streets clustered, against streets to cluster.

        Coarse -- three streets means the bar moves twice -- but the river is
        far the largest, so "2 of 3" is genuinely most of the way. Nothing
        reaches the output directory until the build succeeds, so a street
        boundary is the only moment this work is observable from outside the
        process at all.
        """
        done, total = state.get("done"), state.get("total")
        if not isinstance(done, int | float) or not isinstance(total, int | float) or total <= 0:
            return None
        return Progress(float(done), float(total), self.unit)


def _subject(task: Submission) -> str:
    """What a label names: the run being continued, else the config.

    Run ids are long, share a prefix and differ only at the END, so the middle
    timestamp is dropped -- ``run-production-025433-1095`` -> ``production-1095``.
    """
    if not task.run_id:
        return task.config
    parts = [part for part in task.run_id.removeprefix("run-").split("-") if part]
    return f"{parts[0]}-{parts[-1]}" if len(parts) > 2 else "-".join(parts)


def _flag(flags: object, name: str) -> str:
    """The value of ``--name v`` or ``--name=v`` in a passthrough flag list.

    ``object`` rather than a sequence type: one caller passes a task's own
    flags, the other whatever a task RECORD holds under ``eval_flags``, and a
    record on the share is only ever a claim about its own shape.
    """
    items = list(flags) if isinstance(flags, list | tuple) else []
    for index, item in enumerate(items):
        if item == name:
            return str(items[index + 1]) if index + 1 < len(items) else ""
        if isinstance(item, str) and item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return ""


class VectorSweepTask(TaskKind):
    """Score one CFR kernel against iteration count on one abstraction.

    A measurement, not a run: nothing is trained that anything later consumes,
    and the output is a single JSON curve rather than a checkpoint ladder.

    The parameters ride on fields that already exist rather than on new ones.
    ``config`` is the abstraction directory, ``arm`` is the kernel -- which is
    exactly what an arm IS here, one leg of a comparison -- and ``eval_flags``
    carries the rest verbatim to the command line. Adding six fields to
    TaskSpec, TaskPlan and the environment round-trip to say the same thing
    would widen a wire that three other kinds have to keep passing through.
    """

    name = TaskName.VECTOR_SWEEP
    unit = "checkpoints"
    progress_file = "vector-sweep-progress.json"
    """NOT retried: a retry restarts training from zero rather than resuming, so
    three attempts at a deterministic failure bill three full sweeps."""
    retries = 0

    def validate(self, task: TaskFields) -> None:
        if not task.config:
            raise BadTaskError("a vector-sweep task needs an abstraction directory")

    def commands(self, plan: NodePlan) -> list[list[str]]:
        argv = ["vector-sweep", "--abstraction", plan.config, *plan.eval_flags]
        work = plan.progress_path
        return [[*argv, "--progress-file", work] if work else argv]

    def label(self, task: Submission) -> str:
        return f"vector-{task.arm or 'sweep'}-{task.config}"

    def describe(self, record: Mapping[str, Any]) -> str:
        kernel = record.get("arm") or "sweep"
        return f"vector-sweep {kernel} on {record.get('config') or ''}".strip()

    def sample(self, plan: NodePlan, state: Mapping[str, object]) -> Progress | None:  # noqa: ARG002
        """Checkpoints scored, against checkpoints requested.

        The curve IS the deliverable, so a checkpoint is the honest unit: each
        one is a point that will survive a kill, not a fraction of a single
        answer that only exists at the end.
        """
        done, total = state.get("done"), state.get("total")
        if not isinstance(done, int | float) or not isinstance(total, int | float) or total <= 0:
            return None
        return Progress(float(done), float(total), self.unit)


def samples(rows: Sequence[Mapping[str, Any]], name: str) -> list[Sample]:
    """Finished tasks of one kind, as throughput observations.

    Only tasks that COMPLETED: a task killed at 40% took the wall clock of a
    partial job and would drag every later estimate down. Only tasks that
    recorded units, which excludes everything written before the record carried
    them -- there is no way to reconstruct what those achieved.
    """
    known = kind_of(name)
    found = []
    for row in rows:
        if row.get("op") != name or row.get("cause") != "completed":
            continue
        # A row counted in a unit this kind no longer uses is a DIFFERENT
        # measurement, not an old one. `evaluate` moved from rungs to flop
        # branches; averaging a rung-rate into a branch-rate predicts ~30x wrong
        # and never fails. An absent unit is legacy and taken at face value --
        # those rows predate the field, and no kind whose unit has changed has
        # any of them.
        recorded = row.get("units_unit") or ""
        if known is not None and recorded and recorded != known.unit:
            continue
        seconds = _seconds_between(row.get("started_at"), row.get("ended_at"))
        units = row.get("units") or 0
        if seconds > 0 and units > 0:
            found.append(Sample(float(units), seconds, int(row.get("workers") or 0)))
    return found


def _seconds_between(start: object, end: object) -> float:
    """Two ISO stamps to a duration, or 0 for anything unreadable."""
    try:
        return (
            datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
        ).total_seconds()
    except (TypeError, ValueError):
        return 0.0


def remaining(
    row: Mapping[str, Any], history: Sequence[Mapping[str, Any]], now: str
) -> float | None:
    """Seconds left on a RUNNING task, or None when nothing can say.

    The one place a caller needs: it finds the kind, reads the task's own
    progress, builds the history that kind ran at this width, and asks.
    """
    found = kind_of(row.get("op"))
    if found is None or row.get("cause") in {
        "completed",
        "failed",
        "timeout",
        "killed",
        "cancelled",
    }:
        return None
    elapsed = _seconds_between(row.get("started_at"), now)
    return found.estimate(
        Progress.from_record(row.get("progress")),
        elapsed,
        samples(history, str(row.get("op") or "")),
        int(row.get("workers") or 0),
    )


"""Every kind, by its wire name. Spelled out rather than auto-registered: the
set is closed by :class:`TaskName` anyway, so a hook that discovered subclasses
saved nothing and hid where the list was."""
KINDS: dict[str, TaskKind] = {
    str(instance.name): instance
    for instance in (
        TrainTask(),
        TrainVectorTask(),
        EvaluateTask(),
        PrecomputeTask(),
        VectorSweepTask(),
    )
}


def kind(name: object) -> TaskKind:
    """The kind by name, refusing anything the submit path cannot run."""
    found = KINDS.get(str(name or ""))
    if found is None:
        raise BadTaskError(f"Unknown task kind {name!r}. Known: {', '.join(sorted(KINDS))}.")
    return found


def kind_of(name: object) -> TaskKind | None:
    """The kind, or ``None`` for one this code no longer defines.

    The READ path: listing history must not raise on its own past. The set of
    live kinds has shrunk before -- `vector-sweep` and `train-vector` were both
    retired and both came back -- and a reader that raised on a name it had
    dropped would make the task log unreadable retroactively.
    """
    return KINDS.get(str(name or ""))


def describe(record: Mapping[str, Any]) -> str:
    """One phrase for any task record, including kinds this code no longer has,
    which degrade to their bare op."""
    op = str(record.get("op") or "")
    known = kind_of(op)
    return known.describe(record) if known is not None else op
