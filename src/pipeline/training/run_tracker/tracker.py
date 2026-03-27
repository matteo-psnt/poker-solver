"""Writing the run record as the run progresses, and refusing unsafe resumes."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.pipeline.training.run_tracker.metadata import RunMetadata
from src.shared.config import Config


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
                experiment_id=experiment_id,
                arm=arm,
                parent_run_id=parent_run_id,
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

        return sorted(
            item.name
            for item in base_path.iterdir()
            if item.is_dir() and not item.name.startswith(".") and (item / ".run.json").exists()
        )
