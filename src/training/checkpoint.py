"""
Checkpoint management for training runs.

Handles saving and loading solver checkpoints, including storage state,
iteration count, and configuration.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.solver.base import BaseSolver


@dataclass
class CheckpointInfo:
    """
    Metadata about a checkpoint.

    Attributes:
        iteration: Iteration number when checkpoint was created
        timestamp: ISO timestamp of checkpoint creation
        num_infosets: Number of infosets at checkpoint time
        config_name: Name of configuration used
        checkpoint_dir: Directory containing checkpoint
    """

    iteration: int
    timestamp: str
    num_infosets: int
    config_name: Optional[str] = None
    checkpoint_dir: Optional[Path] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "num_infosets": self.num_infosets,
            "config_name": self.config_name,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointInfo":
        """Create from dictionary."""
        checkpoint_dir = data.get("checkpoint_dir")
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)

        return cls(
            iteration=data["iteration"],
            timestamp=data["timestamp"],
            num_infosets=data["num_infosets"],
            config_name=data.get("config_name"),
            checkpoint_dir=checkpoint_dir,
        )


class CheckpointManager:
    """
    Manages solver checkpoints.

    Handles saving solver state, loading checkpoints, and maintaining
    checkpoint metadata.
    """

    def __init__(self, checkpoint_dir: Path, config_name: Optional[str] = None, run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory to store checkpoints
            config_name: Optional name for this training run
            run_id: Optional unique identifier for this run (auto-generated if not provided)
        """
        self.base_checkpoint_dir = Path(checkpoint_dir)
        self.config_name = config_name or "default"

        # Generate unique run ID if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"
        else:
            self.run_id = run_id

        # Create run-specific subdirectory
        self.checkpoint_dir = self.base_checkpoint_dir / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, solver: BaseSolver, iteration: int) -> Path:
        """
        Save a checkpoint.

        Args:
            solver: Solver to checkpoint
            iteration: Current iteration number

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint info
        info = CheckpointInfo(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            num_infosets=solver.num_infosets(),
            config_name=f"{self.config_name}_{self.run_id}",
            checkpoint_dir=self.checkpoint_dir,
        )

        # Save metadata
        metadata_path = self.checkpoint_dir / f"checkpoint_{iteration}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(info.to_dict(), f, indent=2)

        # Trigger solver's storage checkpoint
        solver.storage.checkpoint(iteration)

        return self.checkpoint_dir

    def load_latest(self) -> Optional[CheckpointInfo]:
        """
        Load metadata for the latest checkpoint.

        Returns:
            CheckpointInfo for latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Return most recent
        return max(checkpoints, key=lambda c: c.iteration)

    def load_checkpoint(self, iteration: int) -> Optional[CheckpointInfo]:
        """
        Load metadata for a specific checkpoint.

        Args:
            iteration: Iteration number of checkpoint

        Returns:
            CheckpointInfo if found, else None
        """
        metadata_path = self.checkpoint_dir / f"checkpoint_{iteration}_metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return CheckpointInfo.from_dict(data)

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """
        List all available checkpoints.

        Returns:
            List of CheckpointInfo objects, sorted by iteration
        """
        checkpoints = []

        for metadata_file in self.checkpoint_dir.glob("checkpoint_*_metadata.json"):
            with open(metadata_file, "r") as f:
                data = json.load(f)

            checkpoints.append(CheckpointInfo.from_dict(data))

        return sorted(checkpoints, key=lambda c: c.iteration)

    def delete_checkpoint(self, iteration: int):
        """
        Delete a specific checkpoint.

        Args:
            iteration: Iteration number of checkpoint to delete
        """
        metadata_path = self.checkpoint_dir / f"checkpoint_{iteration}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

    def clean_old_checkpoints(self, keep_last_n: int = 5):
        """
        Delete old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last_n:
            return

        # Delete all but the last N
        for checkpoint in checkpoints[:-keep_last_n]:
            self.delete_checkpoint(checkpoint.iteration)

    @classmethod
    def list_runs(cls, base_checkpoint_dir: Path) -> list[str]:
        """
        List all training runs in the checkpoint directory.

        Args:
            base_checkpoint_dir: Base checkpoint directory

        Returns:
            List of run IDs
        """
        base_path = Path(base_checkpoint_dir)
        if not base_path.exists():
            return []

        runs = []
        for run_dir in base_path.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                runs.append(run_dir.name)

        return sorted(runs)

    @classmethod
    def from_run_id(cls, base_checkpoint_dir: Path, run_id: str, config_name: Optional[str] = None) -> "CheckpointManager":
        """
        Create CheckpointManager for an existing run.

        Args:
            base_checkpoint_dir: Base checkpoint directory
            run_id: Existing run ID to load
            config_name: Optional config name

        Returns:
            CheckpointManager for the specified run
        """
        return cls(base_checkpoint_dir, config_name=config_name, run_id=run_id)

    def __str__(self) -> str:
        """String representation."""
        num_checkpoints = len(self.list_checkpoints())
        return f"CheckpointManager(run={self.run_id}, dir={self.checkpoint_dir}, checkpoints={num_checkpoints})"
