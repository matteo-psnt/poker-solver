"""Writing the run record as the run progresses, and refusing unsafe resumes."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.pipeline.training.run_tracker.metadata import RunMetadata
from src.shared import records, run_events

if TYPE_CHECKING:
    from src.pipeline.training.run_tracker.attempts import AttemptRecord
    from src.shared.config import Config


class RunTracker:
    """
    Tracks a single training run.

    Appends what happened to ``run_dir/run.jsonl``; the current state is the
    fold of that log. Nothing is rewritten, so there is no window in which a
    kill can leave the record torn -- the failure mode that used to strand a run
    whose checkpoints were fine.
    """

    def __init__(
        self,
        run_dir: Path,
        config_name: str = "default",
        config: Config | None = None,
        action_config_hash: str | None = None,
        card_abstraction_hash: str | None = None,
        experiment_id: str | None = None,
        arm: str | None = None,
        parent_run_id: str | None = None,
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
            experiment_id: Experiment this run is an arm of, if any
            arm: Which arm — e.g. ``"control"`` or ``"variant:<idea>"``. A variant's
                score is uninterpretable without its paired control
            parent_run_id: Base run this was forked from, for base-fork lineage
        """
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.metadata_file = run_events.log_path(self.run_dir)
        self._initialized = False

        # Load existing or prepare new metadata
        if _has_run_record(self.run_dir):
            # Loading an existing run, in either layout.
            self.metadata = RunMetadata.load(self.run_dir)
            self._initialized = True
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
                experiment_id=experiment_id,
                arm=arm,
                parent_run_id=parent_run_id,
            )

    @property
    def metadata_path(self) -> Path:
        return self.metadata_file

    def initialize(self) -> None:
        """Create run directory and initial metadata file.

        Called when training actually starts, not during construction.
        This prevents creating directories for runs that fail during setup.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        events = run_events.read(self.run_dir)

        # Guard on the CREATED event, not on the log being empty and not on a
        # flag: training appends checkpoints to this log before the tracker is
        # initialised, and a run loaded from the pre-log layout has no log at
        # all until now. Either way the fold requires this event to exist.
        if not run_events.head(events):
            run_events.append(self.run_dir, run_events.CREATED, **self.metadata.creation_facts())

        # One `attempt_started` per attempt the metadata knows about. Emitting
        # unconditionally re-opened an attempt every time a tracker was built;
        # skipping it whenever the tracker looked initialised left a resumed
        # run's new attempt unrecorded, so the fold lost it.
        announced = len(run_events.events_of(events, run_events.ATTEMPT_STARTED))
        for attempt in self.metadata.attempts[announced:]:
            self._emit_attempt_started(attempt)
            # And CLOSE it if it already closed. Replaying a pre-log run emitted
            # only the openings, so its finished attempts folded back as
            # `running` with no runtime -- a legacy run resumed by a Batch retry
            # (which never runs `ledger --migrate` first) lost its whole history
            # at exactly the moment the log became the sole source of truth.
            #
            # An attempt killed mid-flight keeps `status="running"` and a null
            # `ended_at`; that dangling shape IS the signal it died, so it must
            # stay open rather than be closed with an invented timestamp.
            if attempt.ended_at is not None or attempt.status != "running":
                self._emit_attempt_ended(attempt)

    def update(
        self,
        iterations: int,
        runtime_seconds: float,
        num_infosets: int,
        storage_capacity: int,
    ) -> None:
        """Update training progress."""
        self.initialize()  # Ensure directory exists
        self.metadata.update_progress(
            iterations=iterations,
            runtime_seconds=runtime_seconds,
            num_infosets=num_infosets,
            storage_capacity=storage_capacity,
        )
        run_events.append(
            self.run_dir,
            run_events.PROGRESS,
            ts=datetime.now(UTC).isoformat(),
            iterations=iterations,
            num_infosets=num_infosets,
            storage_capacity=storage_capacity,
            attempt_runtime_seconds=runtime_seconds,
        )

    def mark_resumed(self) -> None:
        """Open a new attempt: this process is continuing an existing run.

        ``metadata.mark_resumed`` REAPS a still-"running" previous attempt as
        died -- its process is gone or we would not be resuming. That closure is
        a fact the log has to carry: emitted only as `attempt_started`, the fold
        left every reaped attempt "running" forever, which is the exact symptom
        the reaping exists to fix, and dropped its runtime so a run whose tasks
        were all OOM-killed reported ~0s of compute.
        """
        before = len(self.metadata.attempts)
        self.metadata.mark_resumed()
        reaped = [
            a for a in self.metadata.attempts[:before] if a.status != "running" and a.ended_at
        ]
        if reaped:
            # Before initialize(), so the reap is ordered ahead of the attempt
            # that replaced it.
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if run_events.head(run_events.read(self.run_dir)):
                self._emit_attempt_ended(reaped[-1])
        # Announces `created` if missing and any attempt not yet in the log --
        # which on a run converted from the old layout is all of them.
        self.initialize()

    def mark_completed(self) -> None:
        """Mark run as completed."""
        self.initialize()  # Ensure directory exists
        self.metadata.mark_completed()
        self._emit_status("completed")

    def mark_interrupted(self) -> None:
        """Mark run as interrupted by user."""
        self.initialize()  # Ensure directory exists
        self.metadata.mark_interrupted()
        self._emit_status("interrupted")

    def mark_failed(self, cleanup_if_empty: bool = True) -> None:
        """Mark run as failed.

        Args:
            cleanup_if_empty: If True, deletes the run directory if no iterations completed
        """
        if cleanup_if_empty and self.metadata.iterations == 0 and not self._initialized:
            # Failed before any training - don't create directory at all
            return

        self.initialize()  # Ensure directory exists
        self.metadata.mark_failed()
        self._emit_status("failed")

        # Optionally cleanup failed runs with no progress
        if cleanup_if_empty and self.metadata.iterations == 0 and self.run_dir.exists():
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

    def _emit_attempt_started(self, attempt: AttemptRecord | None = None) -> None:
        attempt = attempt or self.metadata.current_attempt
        run_events.append(
            self.run_dir,
            run_events.ATTEMPT_STARTED,
            ts=attempt.started_at,
            index=attempt.index,
            kind=attempt.kind,
            start_iter=attempt.start_iter,
            git_commit=attempt.git_commit,
            git_dirty=attempt.git_dirty,
            git_branch=attempt.git_branch,
            code_snapshot=attempt.code_snapshot,
        )

    def _emit_attempt_ended(self, attempt: AttemptRecord) -> None:
        run_events.append(
            self.run_dir,
            run_events.ATTEMPT_ENDED,
            ts=attempt.ended_at or datetime.now(UTC).isoformat(),
            index=attempt.index,
            end_iter=attempt.end_iter,
            runtime_seconds=attempt.runtime_seconds,
            status=attempt.status,
        )

    def _emit_status(self, status: str) -> None:
        """Close the live attempt and record the run's terminal state."""
        self._emit_attempt_ended(self.metadata.current_attempt)
        run_events.append(
            self.run_dir,
            run_events.STATUS,
            ts=datetime.now(UTC).isoformat(),
            status=status,
            completed_at=self.metadata.completed_at,
            iterations=self.metadata.iterations,
        )

    @classmethod
    def load(cls, run_dir: Path) -> RunTracker:
        """Load existing run tracker."""
        run_path = Path(run_dir)
        if not _has_run_record(run_path):
            raise FileNotFoundError(f"No run record in {run_path}")
        return cls(run_path)

    @staticmethod
    def list_runs(base_dir: Path) -> list[str]:
        """List all runs in directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        return sorted(
            item.name
            for item in base_path.iterdir()
            if item.is_dir() and not item.name.startswith(".") and _has_run_record(item)
        )


def _has_run_record(run_dir: Path) -> bool:
    """Whether this directory holds a run, in either layout.

    ``.run.json`` still counts. Every run written before the event log has one
    and nothing else, and treating those as "not a run" does not merely hide
    them: it makes a resume mint fresh metadata over a live ladder and train
    from zero.
    """
    directory = Path(run_dir)
    return run_events.log_path(directory).exists() or (directory / ".run.json").exists()


def migrate_run_log(run_dir: Path) -> bool:
    """Convert a ``.run.json`` (+ ``progress.jsonl``) run into an event log.

    Replays the snapshot as the events that would have produced it: ``created``
    from the fields fixed at construction, a started/ended pair per recorded
    attempt, a ``checkpoint`` per progress row, and a terminal ``status``. The
    fold of that log is the state the snapshot held.

    Non-destructive and idempotent -- the originals stay for an operator to
    delete, and a directory that already has a log is left alone. Returns
    whether anything was written.
    """
    directory = Path(run_dir)
    if run_events.log_path(directory).exists():
        return False
    snapshot = records.read_snapshot(directory / ".run.json")
    if snapshot is None:
        return False

    metadata = RunMetadata.from_dict(snapshot)
    run_events.append(directory, run_events.CREATED, **metadata.creation_facts())
    for attempt in metadata.attempts:
        run_events.append(
            directory,
            run_events.ATTEMPT_STARTED,
            ts=attempt.started_at,
            index=attempt.index,
            kind=attempt.kind,
            start_iter=attempt.start_iter,
            git_commit=attempt.git_commit,
            git_dirty=attempt.git_dirty,
            git_branch=attempt.git_branch,
            code_snapshot=attempt.code_snapshot,
        )
        if attempt.ended_at is not None or attempt.status != "running":
            run_events.append(
                directory,
                run_events.ATTEMPT_ENDED,
                ts=attempt.ended_at,
                index=attempt.index,
                end_iter=attempt.end_iter,
                runtime_seconds=attempt.runtime_seconds,
                status=attempt.status,
            )

    # The per-checkpoint series, folded in from the file it used to live in.
    for row in records.read_log(directory / "progress.jsonl"):
        run_events.append(
            directory,
            run_events.CHECKPOINT,
            **{k: v for k, v in row.items() if k != "schema_version"},
        )

    run_events.append(
        directory,
        run_events.PROGRESS,
        ts=metadata.started_at,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        storage_capacity=metadata.storage_capacity,
        attempt_runtime_seconds=metadata.current_attempt.runtime_seconds
        if metadata.attempts
        else 0.0,
    )
    if metadata.status != "running":
        run_events.append(
            directory,
            run_events.STATUS,
            ts=metadata.completed_at or metadata.started_at,
            status=metadata.status,
            completed_at=metadata.completed_at,
            iterations=metadata.iterations,
        )
    return True
