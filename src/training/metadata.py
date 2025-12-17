"""
Unified metadata system for training runs and checkpoints.

Provides comprehensive metadata collection for reproducibility and analysis.
"""

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information for reproducibility."""

    python_version: str
    platform: str
    platform_version: str
    hostname: str
    cpu_count: int

    @classmethod
    def collect(cls) -> "SystemInfo":
        """Collect current system information."""
        import socket

        return cls(
            python_version=sys.version.split()[0],
            platform=platform.system(),
            platform_version=platform.version(),
            hostname=socket.gethostname(),
            cpu_count=platform.os.cpu_count() or 1,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProvenanceInfo:
    """Provenance information for reproducibility."""

    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    command: Optional[str] = None

    @classmethod
    def collect(cls, command: Optional[str] = None) -> "ProvenanceInfo":
        """
        Collect provenance information.

        Args:
            command: Command used to start training (if available)
        """
        git_commit = None
        git_branch = None
        git_dirty = False

        try:
            # Try to get git commit
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                git_branch = result.stdout.strip()

            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                git_dirty = bool(result.stdout.strip())

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Git not available or timeout
            pass

        return cls(
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            command=command or " ".join(sys.argv),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingStats:
    """Training statistics tracked over the run."""

    total_iterations: int = 0
    total_runtime_seconds: float = 0.0
    iterations_per_second: float = 0.0
    num_infosets: int = 0
    cache_hit_rate: float = 0.0
    avg_traversal_depth: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CheckpointEntry:
    """Metadata for a single checkpoint."""

    iteration: int
    timestamp: str
    num_infosets: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointEntry":
        """Create from dictionary."""
        return cls(
            iteration=data["iteration"],
            timestamp=data["timestamp"],
            num_infosets=data["num_infosets"],
            metrics=data.get("metrics", {}),
            tags=data.get("tags", []),
        )


@dataclass
class CheckpointManifest:
    """Manifest tracking all checkpoints in a run."""

    run_id: str
    checkpoints: List[CheckpointEntry] = field(default_factory=list)
    latest_iteration: int = 0
    best_checkpoint: Optional[Dict[str, Any]] = None

    def add_checkpoint(
        self,
        iteration: int,
        num_infosets: int,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Add a checkpoint to the manifest.

        Args:
            iteration: Iteration number
            num_infosets: Number of infosets at checkpoint
            metrics: Optional metrics for this checkpoint
            tags: Optional tags (e.g., ["milestone", "best"])
        """
        entry = CheckpointEntry(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            num_infosets=num_infosets,
            metrics=metrics or {},
            tags=tags or [],
        )
        self.checkpoints.append(entry)
        self.latest_iteration = iteration

    def get_checkpoint(self, iteration: int) -> Optional[CheckpointEntry]:
        """Get checkpoint entry by iteration."""
        for checkpoint in self.checkpoints:
            if checkpoint.iteration == iteration:
                return checkpoint
        return None

    def get_latest(self) -> Optional[CheckpointEntry]:
        """Get latest checkpoint entry."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.iteration)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "latest_iteration": self.latest_iteration,
            "best_checkpoint": self.best_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointManifest":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            checkpoints=[CheckpointEntry.from_dict(c) for c in data.get("checkpoints", [])],
            latest_iteration=data.get("latest_iteration", 0),
            best_checkpoint=data.get("best_checkpoint"),
        )

    def save(self, filepath: Path):
        """Save manifest to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "CheckpointManifest":
        """Load manifest from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class RunMetadata:
    """Comprehensive metadata for a training run."""

    run_id: str
    config_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"  # "running", "completed", "failed", "interrupted"
    system: Optional[SystemInfo] = None
    config: Optional[Dict[str, Any]] = None
    provenance: Optional[ProvenanceInfo] = None
    statistics: Optional[TrainingStats] = None

    @classmethod
    def create(
        cls,
        run_id: str,
        config_name: str,
        config: Optional[Dict[str, Any]] = None,
        command: Optional[str] = None,
    ) -> "RunMetadata":
        """
        Create new run metadata.

        Args:
            run_id: Unique run identifier
            config_name: Configuration name
            config: Full configuration dictionary
            command: Command used to start training
        """
        return cls(
            run_id=run_id,
            config_name=config_name,
            started_at=datetime.now().isoformat(),
            system=SystemInfo.collect(),
            config=config,
            provenance=ProvenanceInfo.collect(command),
            statistics=TrainingStats(),
        )

    def update_stats(
        self,
        total_iterations: int,
        total_runtime_seconds: float,
        num_infosets: int,
        cache_hit_rate: float = 0.0,
        avg_traversal_depth: float = 0.0,
    ):
        """Update training statistics."""
        if self.statistics is None:
            self.statistics = TrainingStats()

        self.statistics.total_iterations = total_iterations
        self.statistics.total_runtime_seconds = total_runtime_seconds
        self.statistics.iterations_per_second = (
            total_iterations / total_runtime_seconds if total_runtime_seconds > 0 else 0.0
        )
        self.statistics.num_infosets = num_infosets
        self.statistics.cache_hit_rate = cache_hit_rate
        self.statistics.avg_traversal_depth = avg_traversal_depth

    def mark_completed(self):
        """Mark run as completed."""
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()

    def mark_failed(self):
        """Mark run as failed."""
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "system": self.system.to_dict() if self.system else None,
            "config": self.config,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "statistics": self.statistics.to_dict() if self.statistics else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        """Create from dictionary."""
        system_data = data.get("system")
        provenance_data = data.get("provenance")
        stats_data = data.get("statistics")

        return cls(
            run_id=data["run_id"],
            config_name=data["config_name"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=data.get("status", "running"),
            system=SystemInfo(**system_data) if system_data else None,
            config=data.get("config"),
            provenance=ProvenanceInfo(**provenance_data) if provenance_data else None,
            statistics=TrainingStats(**stats_data) if stats_data else None,
        )

    def save(self, filepath: Path):
        """Save metadata to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "RunMetadata":
        """Load metadata from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
