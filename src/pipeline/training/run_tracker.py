"""
Simple training run tracking.

Just saves basic metadata per run - no complex registry or manifests.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.shared.config import Config


@dataclass
class RunMetadata:
    run_id: str
    config_name: str
    started_at: str
    resumed_at: str | None
    completed_at: str | None
    status: str
    iterations: int
    runtime_seconds: float
    num_infosets: int
    storage_capacity: int
    action_config_hash: str
    config: Config

    @classmethod
    def new(
        cls,
        run_id: str,
        config_name: str,
        config: Config,
        action_config_hash: str,
    ) -> "RunMetadata":
        storage_capacity = config.storage.initial_capacity if config else 0
        return cls(
            run_id=run_id,
            config_name=config_name,
            started_at=datetime.now().isoformat(),
            resumed_at=None,
            completed_at=None,
            status="running",
            iterations=0,
            runtime_seconds=0.0,
            num_infosets=0,
            storage_capacity=storage_capacity,
            action_config_hash=action_config_hash,
            config=config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        config_dict = data.get("config")
        if not isinstance(config_dict, dict) or not config_dict:
            raise ValueError("Run metadata missing required config")
        action_config_hash = data.get("action_config_hash")
        if not isinstance(action_config_hash, str) or not action_config_hash:
            raise ValueError("Run metadata missing required action_config_hash")
        config = Config.from_dict(config_dict)
        return cls(
            run_id=data.get("run_id", ""),
            config_name=data.get("config_name", "default"),
            started_at=data.get("started_at", ""),
            resumed_at=data.get("resumed_at"),
            completed_at=data.get("completed_at"),
            status=data.get("status", "unknown"),
            iterations=int(data.get("iterations", 0)),
            runtime_seconds=float(data.get("runtime_seconds", 0.0)),
            num_infosets=int(data.get("num_infosets", 0)),
            storage_capacity=int(data.get("storage_capacity", 0)),
            action_config_hash=action_config_hash,
            config=config,
        )

    @classmethod
    def load(cls, path: Path) -> "RunMetadata":
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
            "resumed_at": self.resumed_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "num_infosets": self.num_infosets,
            "storage_capacity": self.storage_capacity,
            "action_config_hash": self.action_config_hash,
            "config": config_dict,
        }

    def update_progress(
        self,
        iterations: int,
        runtime_seconds: float,
        num_infosets: int,
        storage_capacity: int,
    ) -> None:
        self.iterations = iterations
        self.runtime_seconds = runtime_seconds
        self.num_infosets = num_infosets
        self.storage_capacity = storage_capacity

    def resolve_initial_capacity(self, default_capacity: int) -> int:
        """Return stored capacity if present, otherwise a default."""
        return self.storage_capacity or default_capacity

    def mark_resumed(self) -> None:
        self.resumed_at = datetime.now().isoformat()
        self.status = "running"

    def mark_completed(self) -> None:
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()

    def mark_interrupted(self) -> None:
        self.status = "interrupted"
        self.completed_at = datetime.now().isoformat()

    def mark_failed(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()


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
        config: "Config | None" = None,
        action_config_hash: str | None = None,
    ):
        """
        Initialize run tracker.

        Args:
            run_dir: Directory for this run
            config_name: Name of config used
            config: Configuration object
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
                self.run_id, config_name, config, action_config_hash=action_config_hash
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

    def _save(self):
        """Save metadata to disk."""
        if not self._initialized:
            # Don't save until initialize() is called
            return
        self.metadata.save(self.metadata_file)

    @classmethod
    def load(cls, run_dir: Path) -> "RunTracker":
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
