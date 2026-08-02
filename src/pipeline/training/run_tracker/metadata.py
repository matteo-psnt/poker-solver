"""The `.run.json` record: what a run is, and how far it got."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.training.run_tracker.attempts import (
    AttemptRecord,
    _opt_str,
)
from src.shared import records, run_events
from src.shared.config import Config
from src.shared.gitinfo import get_git_commit, is_git_dirty


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
    # Experiment lineage. A base-fork experiment is a base run plus several arms,
    # and an arm's score is uninterpretable without knowing which experiment it
    # belongs to and whether it is the variant or the control. None on runs that
    # were not launched as part of an experiment, which includes every legacy run.
    experiment_id: str | None = None
    arm: str | None = None
    parent_run_id: str | None = None
    # Exact hash of the resolved config. `config_name` is not identity: it comes
    # from system.config_name inside the YAML, so a run and its override-variant
    # record the same name. None on pre-hash runs.
    config_hash: str | None = None

    def creation_facts(self) -> dict[str, Any]:
        """Everything fixed when the run was created, for the ``created`` event.

        Deliberately the FIRST event: a run listing answers identity, config and
        provenance from one line rather than folding the whole log.
        """
        return {
            "ts": self.started_at,
            "run_id": self.run_id,
            "config_name": self.config_name,
            "started_at": self.started_at,
            "action_config_hash": self.action_config_hash,
            "card_abstraction_hash": self.card_abstraction_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "experiment_id": self.experiment_id,
            "arm": self.arm,
            "parent_run_id": self.parent_run_id,
            "config_hash": self.config_hash,
            "storage_capacity": self.storage_capacity,
            "config": self.config.to_dict(),
        }

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
        experiment_id: str | None = None,
        arm: str | None = None,
        parent_run_id: str | None = None,
    ) -> RunMetadata:
        storage_capacity = config.storage.initial_capacity if config else 0
        now = datetime.now(UTC).isoformat()
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
            experiment_id=experiment_id,
            arm=arm,
            parent_run_id=parent_run_id,
            config_hash=config.content_hash() if config else None,
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
            git_commit=git_commit,
            git_dirty=git_dirty,
            attempts=attempts,
            # All four are absent on every pre-experiment run, so they default to
            # None rather than being required — a legacy run is simply an
            # unaffiliated one, not an unloadable one.
            experiment_id=_opt_str(data.get("experiment_id")),
            arm=_opt_str(data.get("arm")),
            parent_run_id=_opt_str(data.get("parent_run_id")),
            config_hash=_opt_str(data.get("config_hash")),
        )

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> RunMetadata:
        """Fold a run's event log into its current state.

        ``created`` carries everything fixed at construction; the mutable fields
        are whatever the most recent event carrying them said. Attempts are
        rebuilt by pairing ``attempt_started`` with its ``attempt_ended``.
        """
        created = run_events.head(events)
        if not created:
            raise ValueError("run log has no `created` event")

        attempts: list[AttemptRecord] = []
        for started in run_events.events_of(events, run_events.ATTEMPT_STARTED):
            attempts.append(
                AttemptRecord(
                    index=int(started.get("index", len(attempts))),
                    kind=started.get("kind", "fresh"),
                    started_at=started.get("ts", ""),
                    start_iter=int(started.get("start_iter", 0)),
                    git_commit=started.get("git_commit"),
                    git_dirty=started.get("git_dirty"),
                )
            )
        for ended in run_events.events_of(events, run_events.ATTEMPT_ENDED):
            index = int(ended.get("index", -1))
            for attempt in attempts:
                if attempt.index == index:
                    attempt.ended_at = ended.get("ts")
                    attempt.end_iter = ended.get("end_iter")
                    attempt.runtime_seconds = float(ended.get("runtime_seconds", 0.0))
                    attempt.status = ended.get("status", "completed")

        # The live attempt's runtime is reported by `progress`, which lands more
        # often than `attempt_ended` and is the only account a killed leg leaves.
        if attempts and attempts[-1].status == "running":
            # Only the progress events belonging to THIS attempt: an unscoped
            # scan hands a leg that died before its first checkpoint the
            # PREVIOUS leg's runtime.
            since_last_start = events
            for position, event in enumerate(events):
                if event.get(run_events.EVENT_KEY) == run_events.ATTEMPT_STARTED:
                    since_last_start = events[position:]
            attempts[-1].runtime_seconds = float(
                run_events.tail_value(
                    since_last_start, "attempt_runtime_seconds", 0.0, kind=run_events.PROGRESS
                )
            )

        payload = {
            **created,
            "iterations": run_events.tail_value(events, "iterations", 0, kind=run_events.PROGRESS),
            "num_infosets": run_events.tail_value(
                events, "num_infosets", 0, kind=run_events.PROGRESS
            ),
            "storage_capacity": run_events.tail_value(
                events,
                "storage_capacity",
                created.get("storage_capacity", 0),
                kind=run_events.PROGRESS,
            ),
            # Scoped to STATUS: `attempt_ended` carries the ATTEMPT's status,
            # which is a different fact and was being read as the run's.
            # Cumulative across attempts, the same sum _sync_cumulative_runtime
            # keeps live -- it is not a field any single event carries.
            "runtime_seconds": sum(a.runtime_seconds for a in attempts),
            "status": run_events.tail_value(events, "status", "running", kind=run_events.STATUS),
            "completed_at": run_events.tail_value(events, "completed_at", kind=run_events.STATUS),
            "attempts": [a.to_dict() for a in attempts],
        }
        return cls.from_dict(payload)

    @classmethod
    def load(cls, run_dir: Path) -> RunMetadata:
        """Read a run's state by folding its event log.

        Takes the run DIRECTORY, not a file: which files a run keeps is this
        module's business, not its callers'.
        """
        directory = Path(run_dir)
        events = run_events.read(directory)
        if events:
            return cls.from_events(events)

        # Every run directory that existed before the log holds a .run.json and
        # nothing else, and they are still read on every resume, evaluate, curve
        # and report. Reading them here rather than demanding a migration first
        # is not a second format to maintain: it is the input side of the
        # conversion, and it goes when the last snapshot does. Read-only on
        # purpose -- a listing must not rewrite 43 directories as a side effect.
        snapshot = records.read_snapshot(directory / ".run.json")
        if snapshot is not None:
            return cls.from_dict(snapshot)

        raise ValueError(
            f"No run record in {directory}. Any checkpoints there are unaffected -- "
            "the manifest and snapshots are written separately."
        )

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
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "attempts": [a.to_dict() for a in self.attempts],
            "experiment_id": self.experiment_id,
            "arm": self.arm,
            "parent_run_id": self.parent_run_id,
            "config_hash": self.config_hash,
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
                started_at=datetime.now(UTC).isoformat(),
                start_iter=self.iterations,
                git_commit=get_git_commit(),
                git_dirty=is_git_dirty(),
            )
        )

    def _close_current_attempt(self, status: str) -> None:
        attempt = self.current_attempt
        attempt.status = status
        attempt.ended_at = datetime.now(UTC).isoformat()
        if attempt.end_iter is None:
            attempt.end_iter = self.iterations

    def mark_completed(self) -> None:
        self.status = "completed"
        self.completed_at = datetime.now(UTC).isoformat()
        self._close_current_attempt("completed")

    def mark_interrupted(self) -> None:
        self.status = "interrupted"
        self.completed_at = datetime.now(UTC).isoformat()
        self._close_current_attempt("interrupted")

    def mark_failed(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now(UTC).isoformat()
        self._close_current_attempt("failed")
