"""
Simple training run tracking.

Just saves basic metadata per run - no complex registry or manifests.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.pipeline.training.versioning import REPRESENTATION_VERSION
from src.shared.config import Config
from src.shared.gitinfo import get_git_commit, is_git_dirty


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
        )


@dataclass
class RunMetadata:
    run_id: str
    config_name: str
    started_at: str
    completed_at: str | None
    status: str
    iterations: int
    # Cumulative compute time across every attempt (sum of AttemptRecord.runtime_seconds),
    # NOT just the most recent session. Recomputed on each progress update.
    runtime_seconds: float
    num_infosets: int
    storage_capacity: int
    action_config_hash: str
    config: Config
    # Exact config hash of the card abstraction this run trained against. The config
    # *name* is a mutable pointer: recomputing an abstraction reuses the name but
    # produces different buckets, so evaluating by name alone silently rebuckets the
    # checkpoint. None on pre-provenance runs, which cannot be evaluated faithfully.
    card_abstraction_hash: str | None = None
    # Representation/format version this run was produced under. Pre-versioning
    # (legacy) runs have no field and load as 0.
    representation_version: int = REPRESENTATION_VERSION
    # Code provenance: the commit that produced this checkpoint, and whether the
    # working tree had uncommitted changes at start. A bare hash cannot be trusted
    # when dirty, so both are recorded. None on runs trained outside a git checkout
    # or before this field existed.
    git_commit: str | None = None
    git_dirty: bool | None = None
    # Append-only per-session compute records. attempts[0] is the fresh run; each
    # resume appends one. Empty only on malformed/pre-attempts metadata (synthesized
    # on load, see from_dict).
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def current_attempt(self) -> AttemptRecord:
        """The live (most recent) attempt. Callers mutate this on progress/close."""
        return self.attempts[-1]

    @classmethod
    def new(
        cls,
        run_id: str,
        config_name: str,
        config: Config,
        action_config_hash: str,
        card_abstraction_hash: str | None = None,
    ) -> RunMetadata:
        storage_capacity = config.storage.initial_capacity if config else 0
        now = datetime.now().isoformat()
        git_commit = get_git_commit()
        git_dirty = is_git_dirty()
        return cls(
            run_id=run_id,
            config_name=config_name,
            started_at=now,
            completed_at=None,
            status="running",
            iterations=0,
            runtime_seconds=0.0,
            num_infosets=0,
            storage_capacity=storage_capacity,
            action_config_hash=action_config_hash,
            card_abstraction_hash=card_abstraction_hash,
            config=config,
            git_commit=git_commit,
            git_dirty=git_dirty,
            attempts=[
                AttemptRecord(
                    index=0,
                    kind="fresh",
                    started_at=now,
                    start_iter=0,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                )
            ],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        config_dict = data.get("config")
        if not isinstance(config_dict, dict) or not config_dict:
            raise ValueError("Run metadata missing required config")
        action_config_hash = data.get("action_config_hash")
        if not isinstance(action_config_hash, str) or not action_config_hash:
            raise ValueError("Run metadata missing required action_config_hash")
        config = Config.from_persisted_dict(config_dict)
        started_at = data.get("started_at", "")
        completed_at = data.get("completed_at")
        status = data.get("status", "unknown")
        iterations = int(data.get("iterations", 0))
        runtime_seconds = float(data.get("runtime_seconds", 0.0))
        git_commit = data.get("git_commit") if isinstance(data.get("git_commit"), str) else None
        git_dirty = data.get("git_dirty") if isinstance(data.get("git_dirty"), bool) else None

        raw_attempts = data.get("attempts")
        if isinstance(raw_attempts, list) and raw_attempts:
            attempts = [AttemptRecord.from_dict(a) for a in raw_attempts]
        else:
            # Pre-attempts metadata: synthesize a single attempt spanning the whole
            # run so old runs still load and read as a one-session timeline. The
            # original single resumed_at slot can't be split back into distinct
            # sessions, so a resumed legacy run collapses to one attempt (lossy by
            # necessity, not by design).
            attempts = [
                AttemptRecord(
                    index=0,
                    kind="fresh",
                    started_at=started_at,
                    start_iter=0,
                    ended_at=completed_at,
                    end_iter=iterations,
                    runtime_seconds=runtime_seconds,
                    status=status,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                )
            ]

        return cls(
            run_id=data.get("run_id", ""),
            config_name=data.get("config_name", "default"),
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            iterations=iterations,
            runtime_seconds=runtime_seconds,
            num_infosets=int(data.get("num_infosets", 0)),
            storage_capacity=int(data.get("storage_capacity", 0)),
            action_config_hash=action_config_hash,
            # Missing on pre-provenance runs → None; such runs cannot be pinned to the
            # abstraction they trained against, so evaluation must refuse them.
            card_abstraction_hash=(
                data["card_abstraction_hash"]
                if isinstance(data.get("card_abstraction_hash"), str)
                else None
            ),
            config=config,
            # Missing on pre-versioning runs → 0 (legacy), NOT the current default.
            representation_version=int(data.get("representation_version", 0)),
            git_commit=git_commit,
            git_dirty=git_dirty,
            attempts=attempts,
        )

    @classmethod
    def load(cls, path: Path) -> RunMetadata:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        config_dict = self.config.to_dict()
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "num_infosets": self.num_infosets,
            "storage_capacity": self.storage_capacity,
            "action_config_hash": self.action_config_hash,
            "card_abstraction_hash": self.card_abstraction_hash,
            "representation_version": self.representation_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "attempts": [a.to_dict() for a in self.attempts],
            "config": config_dict,
        }

    def _sync_cumulative_runtime(self) -> None:
        """Top-level runtime_seconds is the sum over all attempts, not the last one."""
        self.runtime_seconds = sum(a.runtime_seconds for a in self.attempts)

    def update_progress(
        self,
        iterations: int,
        runtime_seconds: float,
        num_infosets: int,
        storage_capacity: int,
    ) -> None:
        # ``runtime_seconds`` is this session's elapsed wall time (per-process). Store
        # it on the live attempt and refresh the run-level total; ``iterations`` is the
        # cumulative count, so it also marks how far the current attempt has reached.
        self.iterations = iterations
        self.num_infosets = num_infosets
        self.storage_capacity = storage_capacity
        attempt = self.current_attempt
        attempt.runtime_seconds = runtime_seconds
        attempt.end_iter = iterations
        self._sync_cumulative_runtime()

    def resolve_initial_capacity(self, default_capacity: int) -> int:
        """Return stored capacity if present, otherwise a default."""
        return self.storage_capacity or default_capacity

    def mark_resumed(self) -> None:
        # Open a new attempt starting at the checkpoint we're resuming from. Called
        # while self.iterations still holds the checkpoint count, so start_iter is
        # exactly the resume point.
        self.status = "running"
        # A still-"running" previous attempt cannot actually be running: we are
        # resuming, so its process is gone and no mark_* ever ran (guillotine, OOM,
        # SIGKILL). Reap it. Without this every dead attempt stays "running"
        # forever, and a run whose attempts mostly died reads as a run still in
        # flight -- c2ef8c accumulated 15 such attempts, 4 h of wall clock that
        # committed nothing, and none of them were distinguishable from live ones.
        # end_iter falls back to self.iterations, which for an attempt that died
        # before its first checkpoint equals start_iter: committed nothing, stated
        # explicitly rather than left null.
        if self.attempts and self.attempts[-1].status == "running":
            self._close_current_attempt("died")
        self.attempts.append(
            AttemptRecord(
                index=len(self.attempts),
                kind="resume",
                started_at=datetime.now().isoformat(),
                start_iter=self.iterations,
                git_commit=get_git_commit(),
                git_dirty=is_git_dirty(),
            )
        )

    def _close_current_attempt(self, status: str) -> None:
        attempt = self.current_attempt
        attempt.status = status
        attempt.ended_at = datetime.now().isoformat()
        if attempt.end_iter is None:
            attempt.end_iter = self.iterations

    def mark_completed(self) -> None:
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self._close_current_attempt("completed")

    def mark_interrupted(self) -> None:
        self.status = "interrupted"
        self.completed_at = datetime.now().isoformat()
        self._close_current_attempt("interrupted")

    def mark_failed(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
        self._close_current_attempt("failed")


class RunTracker:
    """
    Tracks a single training run.

    Saves minimal metadata to run_dir/.run.json:
    - run_id, config_name
    - start/end times, status
    - iterations, runtime, infosets
    - action_config_hash
    - config (inline)
    """

    def __init__(
        self,
        run_dir: Path,
        config_name: str = "default",
        config: Config | None = None,
        action_config_hash: str | None = None,
        card_abstraction_hash: str | None = None,
    ):
        """
        Initialize run tracker.

        Args:
            run_dir: Directory for this run
            config_name: Name of config used
            config: Configuration object
            action_config_hash: Hash of the action abstraction
            card_abstraction_hash: Exact config hash of the card abstraction being
                trained against, recorded so evaluation can pin it later
        """
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.metadata_file = self.run_dir / ".run.json"
        self._initialized = False

        # Load existing or prepare new metadata
        if self.metadata_file.exists():
            # Loading existing run
            self.metadata = RunMetadata.load(self.metadata_file)
            self._initialized = True
        else:
            # New run
            if config is None:
                raise ValueError("config is required to create a new run tracker")
            if not action_config_hash:
                raise ValueError("action_config_hash is required to create a new run tracker")
            self.metadata = RunMetadata.new(
                self.run_id,
                config_name,
                config,
                action_config_hash=action_config_hash,
                card_abstraction_hash=card_abstraction_hash,
            )

    @property
    def metadata_path(self) -> Path:
        return self.metadata_file

    def initialize(self):
        """Create run directory and initial metadata file.

        Called when training actually starts, not during construction.
        This prevents creating directories for runs that fail during setup.
        """
        if not self._initialized:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self._save()

    def update(
        self,
        iterations: int,
        runtime_seconds: float,
        num_infosets: int,
        storage_capacity: int,
    ):
        """Update training progress."""
        self.initialize()  # Ensure directory exists
        self.metadata.update_progress(
            iterations=iterations,
            runtime_seconds=runtime_seconds,
            num_infosets=num_infosets,
            storage_capacity=storage_capacity,
        )
        self._save()

    def mark_resumed(self):
        """Mark run as resumed (called when loading from checkpoint)."""
        self.metadata.mark_resumed()
        self._save()

    def mark_completed(self):
        """Mark run as completed."""
        self.initialize()  # Ensure directory exists
        self.metadata.mark_completed()
        self._save()

    def mark_interrupted(self):
        """Mark run as interrupted by user."""
        self.initialize()  # Ensure directory exists
        self.metadata.mark_interrupted()
        self._save()

    def mark_failed(self, cleanup_if_empty: bool = True):
        """Mark run as failed.

        Args:
            cleanup_if_empty: If True, deletes the run directory if no iterations completed
        """
        if cleanup_if_empty and self.metadata.iterations == 0 and not self._initialized:
            # Failed before any training - don't create directory at all
            return

        self.initialize()  # Ensure directory exists
        self.metadata.mark_failed()
        self._save()

        # Optionally cleanup failed runs with no progress
        if cleanup_if_empty and self.metadata.iterations == 0:
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir)

    def verify_action_config_hash(self, actual_hash: str) -> None:
        """Ensure action abstraction hash matches run metadata."""
        if self.metadata.action_config_hash != actual_hash:
            raise ValueError(
                "Action abstraction hash does not match run metadata: "
                f"{self.metadata_path}\n"
                f"  expected: {self.metadata.action_config_hash}\n"
                f"  actual:   {actual_hash}"
            )

    def verify_card_abstraction_hash(self, actual_hash: str | None) -> None:
        """Ensure the card abstraction matches the one this run was trained against.

        Resuming under a recomputed abstraction silently rebuckets every existing
        infoset key, corrupting the run with no error.
        """
        if self.metadata.card_abstraction_hash is None:
            return  # Pre-provenance run: nothing recorded to verify against.
        if self.metadata.card_abstraction_hash != actual_hash:
            raise ValueError(
                "Card abstraction hash does not match run metadata: "
                f"{self.metadata_path}\n"
                f"  expected: {self.metadata.card_abstraction_hash}\n"
                f"  actual:   {actual_hash}\n"
                "The abstraction was recomputed since this run was trained; resuming "
                "would rebucket its existing infosets and silently corrupt it."
            )

    def _save(self):
        """Save metadata to disk."""
        if not self._initialized:
            # Don't save until initialize() is called
            return
        self.metadata.save(self.metadata_file)

    @classmethod
    def load(cls, run_dir: Path) -> RunTracker:
        """Load existing run tracker."""
        run_path = Path(run_dir)
        metadata_path = run_path / ".run.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
        return cls(run_path)

    @staticmethod
    def list_runs(base_dir: Path) -> list[str]:
        """List all runs in directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        runs = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                if (item / ".run.json").exists():
                    runs.append(item.name)

        return sorted(runs)
