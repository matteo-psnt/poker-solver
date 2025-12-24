"""
Simple training run tracking.

Just saves basic metadata per run - no complex registry or manifests.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class RunTracker:
    """
    Tracks a single training run.

    Saves minimal metadata to run_dir/.run.json:
    - run_id, config_name
    - start/end times, status
    - iterations, runtime, infosets
    - config (inline)
    """

    def __init__(
        self,
        run_dir: Path,
        config_name: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize run tracker.

        Args:
            run_dir: Directory for this run
            config_name: Name of config used
            config: Full configuration dict
        """
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.metadata_file = self.run_dir / ".run.json"
        self._initialized = False

        # Load existing or prepare new metadata
        if self.metadata_file.exists():
            # Loading existing run
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
            self._initialized = True
        else:
            # New run
            self.metadata = {
                "run_id": self.run_id,
                "config_name": config_name,
                "started_at": datetime.now().isoformat(),
                "resumed_at": None,
                "completed_at": None,
                "status": "running",
                "iterations": 0,
                "runtime_seconds": 0.0,
                "num_infosets": 0,
                "config": config or {},
            }

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
    ):
        """Update training progress."""
        self.initialize()  # Ensure directory exists
        self.metadata["iterations"] = iterations
        self.metadata["runtime_seconds"] = runtime_seconds
        self.metadata["num_infosets"] = num_infosets
        self._save()

    def mark_resumed(self):
        """Mark run as resumed (called when loading from checkpoint)."""
        self.metadata["resumed_at"] = datetime.now().isoformat()
        self.metadata["status"] = "running"
        self._save()

    def mark_completed(self):
        """Mark run as completed."""
        self.initialize()  # Ensure directory exists
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.now().isoformat()
        self._save()

    def mark_failed(self, cleanup_if_empty: bool = True):
        """Mark run as failed.

        Args:
            cleanup_if_empty: If True, deletes the run directory if no iterations completed
        """
        if cleanup_if_empty and self.metadata.get("iterations", 0) == 0 and not self._initialized:
            # Failed before any training - don't create directory at all
            return

        self.initialize()  # Ensure directory exists
        self.metadata["status"] = "failed"
        self.metadata["completed_at"] = datetime.now().isoformat()
        self._save()

        # Optionally cleanup failed runs with no progress
        if cleanup_if_empty and self.metadata.get("iterations", 0) == 0:
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir)

    def _save(self):
        """Save metadata to disk."""
        if not self._initialized:
            # Don't save until initialize() is called
            return
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, run_dir: Path) -> "RunTracker":
        """Load existing run tracker."""
        tracker = cls(run_dir)
        return tracker

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
