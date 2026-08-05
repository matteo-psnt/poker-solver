"""What KIND of work a task is, and everything that differs because of it.

``op`` was a string, and the behaviour keyed off it lived in four modules: the
argv in ``node.plan``, the label and the submit-time validation in
``cloud.spec``, the retry count in ``cloud.dispatch``, the one-line description
in ``task_log``. Adding a kind meant finding all four, and nothing failed if you
found three -- the missing branch was a task that ran with the wrong argv, or a
retry that billed three full runs to fail three times.

A kind is one class here instead. ``__init_subclass__`` registers it, so there
is no list to keep in step, and ``tests/shared/test_tasks.py`` fails if the
registry and :class:`TaskName` disagree.

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
from typing import Any, ClassVar


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
    EVALUATE = "evaluate"
    PRECOMPUTE = "precompute"


class BadTaskError(ValueError):
    """A task that would waste a node, rejected before it costs one.

    A ``ValueError`` because that is what it is, and because the submit path
    already treats one as a refusal rather than a crash.
    """


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
    def from_record(raw: Any) -> Progress | None:
        """Tolerant: read back from a share record an older wrapper may not have
        written, or may have been killed halfway through writing."""
        if not isinstance(raw, Mapping):
            return None
        try:
            return Progress(float(raw["done"]), float(raw["total"]), str(raw.get("unit") or ""))
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

    KINDS: ClassVar[dict[str, TaskKind]] = {}

    name: ClassVar[TaskName]
    unit: ClassVar[str]
    """A file this kind writes its own progress into, relative to the node's
    scratch. Empty when the work is observable some other way -- training
    through its checkpoint manifest -- and it is what the wrapper branches on,
    so no caller has to know one kind's name."""
    progress_file: ClassVar[str] = ""
    """Batch retries. Work cheap to repeat wants them -- training resumes from
    its last published rung, scoring is idempotent. Work with no
    partial-progress marker does not."""
    retries: ClassVar[int] = 3

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is not None:
            TaskKind.KINDS[str(cls.name)] = cls()

    @abc.abstractmethod
    def validate(self, task: Any) -> None:
        """Reject a submission that cannot run, before it reaches a node."""

    @abc.abstractmethod
    def commands(self, plan: Any) -> list[list[str]]:
        """The `poker-solver` argv(s) this task runs, in order.

        A list because scoring a ladder is one command per rung; the node runs
        them in sequence and accounts for each.
        """

    @abc.abstractmethod
    def label(self, task: Any) -> str:
        """The words a task id is built from -- what this task DOES."""

    @abc.abstractmethod
    def describe(self, record: Mapping[str, Any]) -> str:
        """One phrase saying what this task did, from its recorded fields."""

    @abc.abstractmethod
    def sample(self, plan: Any, state: Mapping[str, Any]) -> Progress | None:
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

    def validate(self, task: Any) -> None:
        if not task.config:
            # A CONTINUING task needs it too: the config builds the tree and the
            # solver, and the checkpoint stores neither. `--run x` without one
            # reached the node and died on `Config file not found` -- after a
            # snapshot upload, a pool spin-up and every retry.
            raise BadTaskError("a training task needs a config, even when continuing a run")
        if task.to <= 0:
            raise BadTaskError("the iteration target is ABSOLUTE and must be positive")

    def commands(self, plan: Any) -> list[list[str]]:
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

    def label(self, task: Any) -> str:
        words = ["train", _subject(task)]
        if task.to:
            words.append(f"to{compact(task.to)}")
        if task.arm:
            words.append(task.arm)
        return "-".join(word for word in words if word)

    def describe(self, record: Mapping[str, Any]) -> str:
        target = str(record.get("target_iteration") or "")
        return f"train ->{compact(int(target))}" if target.isdigit() and target != "0" else "train"

    def sample(self, plan: Any, state: Mapping[str, Any]) -> Progress | None:
        """Iterations done against the target, from the checkpoint manifest.

        The one kind that can already answer, because training writes a
        checkpoint as it goes and the manifest names its iteration.
        """
        done = state.get("iteration")
        if not isinstance(done, int | float) or plan.to <= 0:
            return None
        return Progress(float(done), float(plan.to), self.unit)


class EvaluateTask(TaskKind):
    """Score published rungs of an existing run."""

    name = TaskName.EVALUATE
    unit = "rungs"

    def validate(self, task: Any) -> None:
        if not task.run_id:
            raise BadTaskError("an evaluation works on an existing run, so it needs a run id")

    def commands(self, plan: Any) -> list[list[str]]:
        """One command per rung. ``eval_flags`` is the submitter's passthrough
        (``score --run r -- --br-flops 8``), validated where it is built because
        its contents are unknowable here."""
        commands = []
        for rung in list(plan.eval_rungs) or [""]:
            argv = ["evaluate", "--run", plan.run_id, "--method", plan.eval_method]
            if rung:
                argv += ["--at", rung]
            commands.append(argv + list(plan.eval_flags))
        return commands

    def label(self, task: Any) -> str:
        words = ["score", _subject(task)]
        rung = str(getattr(task, "eval_at", "") or "")
        if rung.isdigit():
            words.append(compact(int(rung)))
        seed = _flag(getattr(task, "eval_flags", ()), "--br-board-seed")
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

    def sample(self, plan: Any, state: Mapping[str, Any]) -> Progress | None:
        """Rungs scored against rungs requested.

        Counted from ZERO rather than reported only once the first rung lands.
        A bar at 0 of 1 looks like nothing is happening, and for a single-rung
        evaluation it will sit there until the end -- but it is what carries the
        TOTAL, and the total is what lets an estimate scale a known rate to the
        work actually asked for. Without it an evaluation can only be predicted
        by the median duration of past ones, which is wrong the moment somebody
        scores ten rungs instead of one.

        Inside a single rung is still opaque. The exact-BR walk is recursive
        over a public tree rather than a flat loop over sampled flops, so a
        counter would have to live in the hot path -- a real cost for a bar.
        """
        total = len(plan.eval_rungs)
        if total <= 0:
            return None
        scored = state.get("scored")
        return Progress(float(scored if isinstance(scored, int) else 0), float(total), self.unit)


class PrecomputeTask(TaskKind):
    """Build a card abstraction."""

    name = TaskName.PRECOMPUTE
    unit = "streets"
    """NOT retried. A precompute has no partial-progress marker -- `metadata.json`
    is written only on success -- so a retry restarts the whole enumeration, and
    a deterministic failure would bill three full runs to fail three times."""
    retries = 0

    def validate(self, task: Any) -> None:
        if not task.config:
            raise BadTaskError("a precompute task needs an abstraction config")

    progress_file = "precompute-progress.json"

    def commands(self, plan: Any) -> list[list[str]]:
        argv = ["precompute", "--config", plan.config, "--json"]
        # Where the build reports street completion, and where `sample` reads it
        # back. Node-local: it is a heartbeat, not a record.
        work = getattr(plan, "progress_path", "")
        return [[*argv, "--progress-file", str(work)] if work else argv]

    def label(self, task: Any) -> str:
        return f"precompute-{task.config}"

    def describe(self, record: Mapping[str, Any]) -> str:
        return f"precompute {record.get('config') or ''}".strip()

    def sample(self, plan: Any, state: Mapping[str, Any]) -> Progress | None:  # noqa: ARG002
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


def _subject(task: Any) -> str:
    """What a label names: the run being continued, else the config.

    Run ids are long, share a prefix and differ only at the END, so the middle
    timestamp is dropped -- ``run-production-025433-1095`` -> ``production-1095``.
    """
    run_id = getattr(task, "run_id", "") or ""
    if not run_id:
        return getattr(task, "config", "") or ""
    parts = [part for part in run_id.removeprefix("run-").split("-") if part]
    return f"{parts[0]}-{parts[-1]}" if len(parts) > 2 else "-".join(parts)


def _flag(flags: Any, name: str) -> str:
    """The value of ``--name v`` or ``--name=v`` in a passthrough flag list."""
    items = list(flags) if isinstance(flags, list | tuple) else []
    for index, item in enumerate(items):
        if item == name:
            return str(items[index + 1]) if index + 1 < len(items) else ""
        if isinstance(item, str) and item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return ""


def samples(rows: Sequence[Mapping[str, Any]], name: str) -> list[Sample]:
    """Finished tasks of one kind, as throughput observations.

    Only tasks that COMPLETED: a task killed at 40% took the wall clock of a
    partial job and would drag every later estimate down. Only tasks that
    recorded units, which excludes everything written before the record carried
    them -- there is no way to reconstruct what those achieved.
    """
    found = []
    for row in rows:
        if row.get("op") != name or row.get("cause") != "completed":
            continue
        seconds = _seconds_between(row.get("started_at"), row.get("ended_at"))
        units = row.get("units") or 0
        if seconds > 0 and units > 0:
            found.append(Sample(float(units), seconds, int(row.get("workers") or 0)))
    return found


def _seconds_between(start: Any, end: Any) -> float:
    """Two ISO stamps to a duration, or 0 for anything unreadable."""
    try:
        return (
            datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
        ).total_seconds()
    except (TypeError, ValueError):
        return 0.0


def remaining(
    row: Mapping[str, Any], history: Sequence[Mapping[str, Any]], now: Any
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


def kind(name: Any) -> TaskKind:
    """The kind by name, refusing anything the submit path cannot run."""
    found = TaskKind.KINDS.get(str(name or ""))
    if found is None:
        known = ", ".join(sorted(TaskKind.KINDS))
        raise BadTaskError(f"Unknown task kind {name!r}. Known: {known}.")
    return found


def kind_of(name: Any) -> TaskKind | None:
    """The kind, or ``None`` for one this code no longer defines.

    The READ path. The task log holds `vector-sweep` and `train-vector` from
    deleted work, and listing history must not raise on its own past.
    """
    return TaskKind.KINDS.get(str(name or ""))


def describe(record: Mapping[str, Any]) -> str:
    """One phrase for any task record, including kinds this code no longer has,
    which degrade to their bare op."""
    op = str(record.get("op") or "")
    known = kind_of(op)
    return known.describe(record) if known is not None else op
