"""
Training run management with rich metadata and run tracking.

Provides a TrainingRun class that represents a single training session,
with snapshot tracking, metrics, and metadata collection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.solver.base import BaseSolver
from src.training.metadata import (
    CheckpointEntry,
    ProvenanceInfo,
    RunMetadata,
    SystemInfo,
    TrainingStats,
)

logger = logging.getLogger(__name__)


class SnapshotManifest:
    """Manifest tracking all solver snapshots in a run."""

    def __init__(self, run_id: str):
        """
        Initialize snapshot manifest.

        Args:
            run_id: Unique run identifier
        """
        self.run_id = run_id
        self.snapshots: List[CheckpointEntry] = []
        self.latest_iteration: int = 0
        self.best_snapshot: Optional[Dict[str, Any]] = None

    def add_snapshot(
        self,
        iteration: int,
        num_infosets: int,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Add a snapshot to the manifest.

        Args:
            iteration: Iteration number
            num_infosets: Number of infosets at snapshot
            metrics: Optional metrics for this snapshot
            tags: Optional tags (e.g., ["milestone", "best"])
        """
        entry = CheckpointEntry(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            num_infosets=num_infosets,
            metrics=metrics or {},
            tags=tags or [],
        )
        self.snapshots.append(entry)
        self.latest_iteration = iteration

    def get_snapshot(self, iteration: int) -> Optional[CheckpointEntry]:
        """Get snapshot entry by iteration."""
        for snapshot in self.snapshots:
            if snapshot.iteration == iteration:
                return snapshot
        return None

    def get_latest(self) -> Optional[CheckpointEntry]:
        """Get latest snapshot entry."""
        if not self.snapshots:
            return None
        return max(self.snapshots, key=lambda c: c.iteration)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "latest_iteration": self.latest_iteration,
            "best_snapshot": self.best_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotManifest":
        """Create from dictionary."""
        manifest = cls(run_id=data.get("run_id"))
        manifest.snapshots = [CheckpointEntry.from_dict(s) for s in data.get("snapshots", [])]
        manifest.latest_iteration = data.get("latest_iteration", 0)
        manifest.best_snapshot = data.get("best_snapshot")
        return manifest

    def save(self, filepath: Path):
        """Save manifest to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "SnapshotManifest":
        """Load manifest from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ExtendedRunMetadata(RunMetadata):
    """
    Extended metadata for training runs with additional tracking features.
    """

    def __init__(
        self,
        run_id: str,
        config_name: str,
        started_at: str,
        run_name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        abstraction_name: Optional[str] = None,
        abstraction_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize extended run metadata.

        Args:
            run_id: Unique run identifier
            config_name: Configuration name
            started_at: Start timestamp
            run_name: Human-readable run name
            group: Run group (e.g., "baselines", "ablations")
            tags: List of tags for filtering
            description: Run description
            abstraction_name: Name of abstraction used
            abstraction_path: Path to abstraction file
            **kwargs: Additional fields for RunMetadata
        """
        super().__init__(run_id=run_id, config_name=config_name, started_at=started_at, **kwargs)

        self.run_name = run_name or run_id
        self.group = group
        self.tags = tags or []
        self.description = description
        self.abstraction_name = abstraction_name
        self.abstraction_path = abstraction_path

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "run_name": self.run_name,
                "group": self.group,
                "tags": self.tags,
                "description": self.description,
                "abstraction": {
                    "name": self.abstraction_name,
                    "path": self.abstraction_path,
                }
                if self.abstraction_name
                else None,
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> "ExtendedRunMetadata":
        """Create from dictionary."""
        # Extract run-specific fields
        run_name = data.pop("run_name", None)
        group = data.pop("group", None)
        tags = data.pop("tags", [])
        description = data.pop("description", None)

        abstraction = data.pop("abstraction", None)
        abstraction_name = None
        abstraction_path = None
        if abstraction:
            abstraction_name = abstraction.get("name")
            abstraction_path = abstraction.get("path")

        # Extract base fields
        system_data = data.pop("system", None)
        provenance_data = data.pop("provenance", None)
        stats_data = data.pop("statistics", None)

        # Create instance
        run_metadata = cls(
            run_id=data["run_id"],
            config_name=data["config_name"],
            started_at=data["started_at"],
            run_name=run_name,
            group=group,
            tags=tags,
            description=description,
            abstraction_name=abstraction_name,
            abstraction_path=abstraction_path,
            completed_at=data.get("completed_at"),
            status=data.get("status", "running"),
            config=data.get("config"),
        )

        # Restore complex objects
        if system_data:
            run_metadata.system = SystemInfo(**system_data)
        if provenance_data:
            run_metadata.provenance = ProvenanceInfo(**provenance_data)
        if stats_data:
            run_metadata.statistics = TrainingStats(**stats_data)

        return run_metadata

    @classmethod
    def create(  # type: ignore[override]  # Parent class has different signature pattern
        cls,
        run_id: str,
        config_name: str,
        run_name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        abstraction_name: Optional[str] = None,
        abstraction_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        command: Optional[str] = None,
    ) -> "ExtendedRunMetadata":
        """
        Create new extended run metadata.

        Args:
            run_id: Unique run identifier
            config_name: Configuration name
            run_name: Human-readable run name
            group: Run group
            tags: List of tags
            description: Run description
            abstraction_name: Name of abstraction used
            abstraction_path: Path to abstraction file
            config: Full configuration dictionary
            command: Command used to start training
        """
        return cls(
            run_id=run_id,
            config_name=config_name,
            started_at=datetime.now().isoformat(),
            run_name=run_name,
            group=group,
            tags=tags,
            description=description,
            abstraction_name=abstraction_name,
            abstraction_path=abstraction_path,
            system=SystemInfo.collect(),
            config=config,
            provenance=ProvenanceInfo.collect(command),
            statistics=TrainingStats(),
            status="running",
        )


class RunsRegistry:
    """Registry for tracking all training runs."""

    def __init__(self, registry_file: Path):
        """
        Initialize runs registry.

        Args:
            registry_file: Path to registry JSON file
        """
        self.registry_file = Path(registry_file)
        self.runs: List[Dict[str, Any]] = []
        self.groups: Dict[str, List[str]] = {}

        if self.registry_file.exists():
            self.load()

    def load(self):
        """Load registry from disk."""
        try:
            with open(self.registry_file, "r") as f:
                data = json.load(f)

            self.runs = data.get("runs", [])
            self.groups = data.get("groups", {})

        except Exception as e:
            logger.warning(f"Failed to load runs registry: {e}")
            self.runs = []
            self.groups = {}

    def save(self):
        """Save registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "runs": self.runs,
            "groups": self.groups,
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        run_id: str,
        run_name: str,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: str = "running",
        abstraction_name: Optional[str] = None,
        started_at: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Register a new training run."""
        entry = {
            "id": run_id,
            "name": run_name,
            "group": group,
            "tags": tags or [],
            "status": status,
            "abstraction": abstraction_name,
            "started_at": started_at or datetime.now().isoformat(),
            "metrics": metrics or {},
        }

        # Add to runs list
        self.runs.append(entry)

        # Add to group
        if group:
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(run_id)

        self.save()

    def update_status(self, run_id: str, status: str):
        """Update run status."""
        for run in self.runs:
            if run["id"] == run_id:
                run["status"] = status
                self.save()
                break

    def update_metrics(self, run_id: str, metrics: Dict[str, Any]):
        """Update run metrics."""
        for run in self.runs:
            if run["id"] == run_id:
                run["metrics"].update(metrics)
                self.save()
                break

    def list_by_group(self, group: str) -> List[Dict[str, Any]]:
        """List runs in a group."""
        return [run for run in self.runs if run.get("group") == group]

    def list_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """List runs with a tag."""
        return [run for run in self.runs if tag in run.get("tags", [])]


class TrainingRun:
    """
    Represents a single training run with rich metadata and snapshot tracking.

    A TrainingRun encapsulates all aspects of a single solver training session:
    - Snapshots (periodic saves during training)
    - Metrics tracking (exploitability, utility, etc.)
    - Configuration and provenance
    - System information

    Uses:
    - runs.json: Run-level metadata
    - snapshots.json: List of all snapshots with metrics
    - config.yaml: Full configuration
    - solver data files (HDF5, pickle, etc.)
    """

    def __init__(
        self,
        base_dir: Path,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        abstraction_name: Optional[str] = None,
        abstraction_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None,
    ):
        """
        Initialize training run.

        Args:
            base_dir: Base directory to store runs (data/runs)
            run_name: Human-readable run name (optional)
            run_id: Optional unique identifier (auto-generated if not provided)
            group: Run group (e.g., "baselines", "ablations")
            tags: List of tags for filtering (e.g., ["cfr+", "baseline"])
            description: Run description
            abstraction_name: Name of abstraction used
            abstraction_path: Path to abstraction file
            config: Optional full configuration dictionary
            config_name: Optional config name
        """
        self.base_dir = Path(base_dir)
        self.config_name = config_name or "default"

        # Generate unique run ID if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_name:
                self.run_id = f"{run_name}-{timestamp}"
            else:
                self.run_id = f"run-{timestamp}"
        else:
            self.run_id = run_id

        # Run-specific directory
        self.run_dir = self.base_dir / self.run_id

        # Metadata
        self.run_metadata: Optional[ExtendedRunMetadata] = None
        self.snapshot_manifest: Optional[SnapshotManifest] = None

        # Track if we've initialized
        self.initialized = False

        # Store parameters for later
        self._run_name = run_name or self.run_id
        self._group = group
        self._tags = tags or []
        self._description = description
        self._abstraction_name = abstraction_name
        self._abstraction_path = abstraction_path
        self._config = config

        # Initialize registry
        self.registry = RunsRegistry(self.base_dir / ".runs.json")

        # Try to load existing run if directory exists
        if self.run_dir.exists():
            self._load_existing_run()

    def _ensure_initialized(self):
        """Ensure run directory and metadata are initialized."""
        if self.initialized:
            return

        # Create directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory: {self.run_dir}")

        # Create run metadata
        if self.run_metadata is None:
            self.run_metadata = ExtendedRunMetadata.create(
                run_id=self.run_id,
                config_name=self.config_name,
                run_name=self._run_name,
                group=self._group,
                tags=self._tags,
                description=self._description,
                abstraction_name=self._abstraction_name,
                abstraction_path=self._abstraction_path,
                config=self._config,
            )
            self.run_metadata.save(self.run_dir / ".run.json")
            logger.info(f"Created run metadata for {self.run_id}")

        # Save config as YAML file for easy inspection
        if self._config is not None:
            config_file = self.run_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {config_file}")

        # Create snapshot manifest
        if self.snapshot_manifest is None:
            self.snapshot_manifest = SnapshotManifest(run_id=self.run_id)
            self.snapshot_manifest.save(self.run_dir / "snapshots.json")
            logger.info(f"Created snapshot manifest for {self.run_id}")

        # Register in runs registry
        self.registry.register(
            run_id=self.run_id,
            run_name=self._run_name,
            group=self._group,
            tags=self._tags,
            status="running",
            abstraction_name=self._abstraction_name,
            started_at=self.run_metadata.started_at,
        )

        self.initialized = True

    def _load_existing_run(self):
        """Load existing run metadata and manifest."""
        # Try new filename first, fallback to old
        run_metadata_path = self.run_dir / ".run.json"
        snapshot_manifest_path = self.run_dir / "snapshots.json"

        try:
            if run_metadata_path.exists():
                self.run_metadata = ExtendedRunMetadata.load(run_metadata_path)
                logger.info(f"Loaded run metadata for {self.run_id}")

            if snapshot_manifest_path.exists():
                self.snapshot_manifest = SnapshotManifest.load(snapshot_manifest_path)
                logger.info(
                    f"Loaded snapshot manifest: {len(self.snapshot_manifest.snapshots)} snapshots"
                )

            self.initialized = True

        except Exception as e:
            logger.warning(f"Failed to load existing run metadata: {e}")
            self.initialized = False

    def save_snapshot(
        self,
        solver: BaseSolver,
        iteration: int,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Save a snapshot.

        Args:
            solver: Solver to snapshot
            iteration: Current iteration number
            metrics: Optional metrics for this snapshot
            tags: Optional tags (e.g., ["milestone", "best"])

        Returns:
            Path to run directory
        """
        # Ensure initialized
        self._ensure_initialized()

        # Add snapshot to manifest
        num_infosets = solver.num_infosets()
        self.snapshot_manifest.add_snapshot(
            iteration=iteration,
            num_infosets=num_infosets,
            metrics=metrics or {},
            tags=tags or [],
        )

        # Save manifest
        self.snapshot_manifest.save(self.run_dir / "snapshots.json")

        # Update registry metrics
        if metrics:
            self.registry.update_metrics(self.run_id, metrics)

        # Trigger solver's storage snapshot
        solver.storage.checkpoint(iteration)

        logger.info(
            f"Snapshot saved: iteration={iteration}, num_infosets={num_infosets}, "
            f"dir={self.run_dir}"
        )

        return self.run_dir

    def update_stats(
        self,
        total_iterations: int,
        total_runtime_seconds: float,
        num_infosets: int,
        cache_hit_rate: float = 0.0,
        avg_traversal_depth: float = 0.0,
    ):
        """Update training statistics."""
        if not self.initialized or self.run_metadata is None:
            return

        self.run_metadata.update_stats(
            total_iterations=total_iterations,
            total_runtime_seconds=total_runtime_seconds,
            num_infosets=num_infosets,
            cache_hit_rate=cache_hit_rate,
            avg_traversal_depth=avg_traversal_depth,
        )

        # Save updated metadata
        self.run_metadata.save(self.run_dir / ".run.json")

    def mark_completed(self):
        """Mark run as completed."""
        if self.run_metadata is not None:
            self.run_metadata.mark_completed()
            self.run_metadata.save(self.run_dir / ".run.json")
            self.registry.update_status(self.run_id, "completed")
            logger.info(f"Run {self.run_id} marked as completed")

    def mark_failed(self):
        """Mark run as failed."""
        if self.run_metadata is not None:
            self.run_metadata.mark_failed()
            self.run_metadata.save(self.run_dir / ".run.json")
            self.registry.update_status(self.run_id, "failed")
            logger.info(f"Run {self.run_id} marked as failed")

    def get_snapshot(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Get snapshot metadata by iteration."""
        if self.snapshot_manifest is None:
            return None

        entry = self.snapshot_manifest.get_snapshot(iteration)
        return entry.to_dict() if entry else None

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get latest snapshot metadata."""
        if self.snapshot_manifest is None:
            return None

        entry = self.snapshot_manifest.get_latest()
        return entry.to_dict() if entry else None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all snapshots."""
        if self.snapshot_manifest is None:
            return []

        return [s.to_dict() for s in self.snapshot_manifest.snapshots]

    def get_latest_iteration(self) -> int:
        """Get iteration number of latest snapshot."""
        if self.snapshot_manifest is None:
            return 0

        latest = self.snapshot_manifest.get_latest()
        return latest.iteration if latest else 0

    @classmethod
    def list_runs(cls, base_dir: Path) -> List[str]:
        """List all training runs in directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        runs = []
        for run_dir in base_path.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("."):
                # Check if it has metadata
                if (run_dir / ".run.json").exists():
                    runs.append(run_dir.name)

        return sorted(runs)

    @classmethod
    def from_run_id(
        cls,
        base_dir: Path,
        run_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> "TrainingRun":
        """
        Create TrainingRun for an existing run.

        Args:
            base_dir: Base runs directory
            run_id: Existing run ID to load
            config: Optional config dict

        Returns:
            TrainingRun for the specified run

        Raises:
            ValueError: If run doesn't exist
        """
        run_dir = base_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} does not exist in {base_dir}")

        return cls(
            base_dir=base_dir,
            run_id=run_id,
            config=config,
        )

    def __str__(self) -> str:
        """String representation."""
        num_snapshots = len(self.snapshot_manifest.snapshots) if self.snapshot_manifest else 0
        status = "initialized" if self.initialized else "not initialized"
        return (
            f"TrainingRun(id={self.run_id}, name={self._run_name}, "
            f"dir={self.run_dir}, snapshots={num_snapshots}, status={status})"
        )
