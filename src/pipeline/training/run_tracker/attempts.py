"""The small value types a run records: its experiment tag, and one attempt.

An attempt is one process's turn at a run. A run that was resumed four times
has four of them, which is what makes cumulative runtime recoverable after a
crash."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _opt_str(value: Any) -> str | None:
    """A non-empty string, or None — guards against a persisted null or empty string."""
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class ExperimentTag:
    """Which experiment, and which arm of it, a run belongs to.

    Travels together from the CLI down to the run metadata because the three are
    only meaningful as a set: an arm without its experiment cannot be grouped, and
    a variant without a paired control cannot be attributed — the extra training a
    fork receives moves the score on its own.
    """

    experiment_id: str | None = None
    arm: str | None = None
    parent_run_id: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.experiment_id or self.arm or self.parent_run_id)


@dataclass
class AttemptRecord:
    """One contiguous compute session (container lifetime) for a run.

    A run is trained across N attempts: the initial ``fresh`` attempt plus one
    ``resume`` per checkpoint-restart. Recording each attempt separately — rather
    than overwriting a single ``resumed_at``/``runtime_seconds`` slot — is what
    lets the wall-clock timeline (and per-chunk timing of the mandatory <40min
    resume chunks) be reconstructed instead of lost on every restart.

    ``end_iter``/``runtime_seconds`` are refreshed on each checkpoint, so an
    attempt that is killed mid-flight (guillotine, OOM) retains its last
    checkpointed iteration and compute time even though ``mark_*`` never ran and
    its ``status`` stays ``running`` with ``ended_at`` null — that dangling shape
    is itself the signal that the attempt died, to be cross-referenced with the
    client-side orchestration log's Modal exit status.
    """

    index: int
    kind: str  # "fresh" | "resume"
    started_at: str
    start_iter: int
    ended_at: str | None = None
    end_iter: int | None = None
    runtime_seconds: float = 0.0
    status: str = "running"
    git_commit: str | None = None
    git_dirty: bool | None = None
    # The branch this attempt ran from. A run RESUMED from a different
    # worktree is a real case here -- the arms share a commit and differ
    # only in what is uncommitted -- so this is per attempt, not per run.
    git_branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "started_at": self.started_at,
            "start_iter": self.start_iter,
            "ended_at": self.ended_at,
            "end_iter": self.end_iter,
            "runtime_seconds": self.runtime_seconds,
            "status": self.status,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_branch": self.git_branch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptRecord:
        return cls(
            index=int(data.get("index", 0)),
            kind=data.get("kind", "fresh"),
            started_at=data.get("started_at", ""),
            start_iter=int(data.get("start_iter", 0)),
            ended_at=data.get("ended_at"),
            end_iter=(int(data["end_iter"]) if data.get("end_iter") is not None else None),
            runtime_seconds=float(data.get("runtime_seconds", 0.0)),
            status=data.get("status", "unknown"),
            git_commit=data.get("git_commit") if isinstance(data.get("git_commit"), str) else None,
            git_dirty=data.get("git_dirty") if isinstance(data.get("git_dirty"), bool) else None,
            git_branch=(
                data.get("git_branch") if isinstance(data.get("git_branch"), str) else None
            ),
        )
