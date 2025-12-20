"""
Run management for training runs.

Handles saving and loading solver state using unified metadata system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.solver.base import BaseSolver
from src.training.metadata import CheckpointManifest, RunMetadata

logger = logging.getLogger(__name__)


class RunManager:
    """
    Manages training runs with unified metadata.

    Uses:
    - run_metadata.json: Run-level metadata (config, system, provenance)
    - checkpoint_manifest.json: List of all checkpoints with metrics
    - config.yaml: Full configuration (optional, if YAML config provided)
    """

    def __init__(
        self,
        runs_dir: Path,
        config_name: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize run manager.

        Args:
            runs_dir: Base directory to store training runs
            config_name: Optional name for this training run
            run_id: Optional unique identifier (auto-generated if not provided)
            config: Optional full configuration dictionary

        Note:
            Directory is NOT created until first save (lazy creation).
        """
        self.base_runs_dir = Path(runs_dir)
        self.config_name = config_name or "default"

        # Generate unique run ID if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"
        else:
            self.run_id = run_id

        # Run-specific directory (not created yet!)
        self.run_dir = self.base_runs_dir / self.run_id

        # Metadata
        self.run_metadata: Optional[RunMetadata] = None
        self.manifest: Optional[CheckpointManifest] = None

        # Track if we've initialized (saved first checkpoint)
        self.initialized = False

        # Store config for later
        self._config = config

        # Try to load existing run if directory exists
        if self.run_dir.exists():
            self._load_existing_run()

    def _ensure_initialized(self):
        """Ensure checkpoint directory and metadata are initialized."""
        if self.initialized:
            return

        # Create directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory: {self.run_dir}")

        # Create run metadata
        if self.run_metadata is None:
            self.run_metadata = RunMetadata.create(
                run_id=self.run_id,
                config_name=self.config_name,
                config=self._config,
            )
            self.run_metadata.save(self.run_dir / "run_metadata.json")
            logger.info(f"Created run metadata for {self.run_id}")

        # Save config as YAML file for easy inspection
        if self._config is not None:
            config_file = self.run_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {config_file}")

        # Create manifest
        if self.manifest is None:
            self.manifest = CheckpointManifest(run_id=self.run_id)
            self.manifest.save(self.run_dir / "checkpoint_manifest.json")
            logger.info(f"Created checkpoint manifest for {self.run_id}")

        self.initialized = True

    def _load_existing_run(self):
        """Load existing run metadata and manifest."""
        run_metadata_path = self.run_dir / "run_metadata.json"
        manifest_path = self.run_dir / "checkpoint_manifest.json"

        try:
            if run_metadata_path.exists():
                self.run_metadata = RunMetadata.load(run_metadata_path)
                logger.info(f"Loaded run metadata for {self.run_id}")

            if manifest_path.exists():
                self.manifest = CheckpointManifest.load(manifest_path)
                logger.info(
                    f"Loaded checkpoint manifest: {len(self.manifest.checkpoints)} checkpoints"
                )

            self.initialized = True

        except Exception as e:
            logger.warning(f"Failed to load existing run metadata: {e}")
            self.initialized = False

    def save(
        self,
        solver: BaseSolver,
        iteration: int,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            solver: Solver to checkpoint
            iteration: Current iteration number
            metrics: Optional metrics for this checkpoint
            tags: Optional tags (e.g., ["milestone", "best"])

        Returns:
            Path to checkpoint directory
        """
        # Ensure initialized (creates directory on first save)
        self._ensure_initialized()

        # Add checkpoint to manifest
        num_infosets = solver.num_infosets()
        self.manifest.add_checkpoint(
            iteration=iteration,
            num_infosets=num_infosets,
            metrics=metrics or {},
            tags=tags or [],
        )

        # Save manifest
        self.manifest.save(self.run_dir / "checkpoint_manifest.json")

        # Trigger solver's storage checkpoint
        solver.storage.checkpoint(iteration)

        logger.info(
            f"Checkpoint saved: iteration={iteration}, num_infosets={num_infosets}, "
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
        """
        Update training statistics.

        Args:
            total_iterations: Total iterations completed
            total_runtime_seconds: Total training time in seconds
            num_infosets: Current number of infosets
            cache_hit_rate: Cache hit rate (0.0-1.0)
            avg_traversal_depth: Average game tree traversal depth
        """
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
        self.run_metadata.save(self.run_dir / "run_metadata.json")

    def mark_completed(self):
        """Mark training run as completed."""
        if self.run_metadata is not None:
            self.run_metadata.mark_completed()
            self.run_metadata.save(self.run_dir / "run_metadata.json")
            logger.info(f"Run {self.run_id} marked as completed")

    def mark_failed(self):
        """Mark training run as failed."""
        if self.run_metadata is not None:
            self.run_metadata.mark_failed()
            self.run_metadata.save(self.run_dir / "run_metadata.json")
            logger.info(f"Run {self.run_id} marked as failed")

    def get_checkpoint(self, iteration: int) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint metadata by iteration.

        Args:
            iteration: Iteration number

        Returns:
            Checkpoint metadata dict or None
        """
        if self.manifest is None:
            return None

        entry = self.manifest.get_checkpoint(iteration)
        return entry.to_dict() if entry else None

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get latest checkpoint metadata.

        Returns:
            Checkpoint metadata dict or None
        """
        if self.manifest is None:
            return None

        entry = self.manifest.get_latest()
        return entry.to_dict() if entry else None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints.

        Returns:
            List of checkpoint metadata dicts
        """
        if self.manifest is None:
            return []

        return [c.to_dict() for c in self.manifest.checkpoints]

    def get_latest_iteration(self) -> int:
        """
        Get iteration number of latest checkpoint.

        Returns:
            Latest iteration number, or 0 if no checkpoints
        """
        if self.manifest is None:
            return 0

        latest = self.manifest.get_latest()
        return latest.iteration if latest else 0

    @classmethod
    def list_runs(cls, base_runs_dir: Path) -> List[str]:
        """
        List all training runs in runs directory.

        Args:
            base_runs_dir: Base runs directory

        Returns:
            List of run IDs
        """
        base_path = Path(base_runs_dir)
        if not base_path.exists():
            return []

        runs = []
        for run_dir in base_path.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                # Check if it has metadata (new format or old format)
                has_new_format = (run_dir / "run_metadata.json").exists()
                has_old_format = (run_dir / "experiment.json").exists()

                if has_new_format or has_old_format:
                    runs.append(run_dir.name)

        return sorted(runs)

    @classmethod
    def from_run_id(
        cls,
        base_runs_dir: Path,
        run_id: str,
        config_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "RunManager":
        """
        Create RunManager for an existing run.

        Args:
            base_runs_dir: Base runs directory
            run_id: Existing run ID to load
            config_name: Optional config name
            config: Optional config dict

        Returns:
            RunManager for the specified run

        Raises:
            ValueError: If run doesn't exist
        """
        run_dir = base_runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} does not exist in {base_runs_dir}")

        return cls(
            runs_dir=base_runs_dir,
            config_name=config_name,
            run_id=run_id,
            config=config,
        )

    def __str__(self) -> str:
        """String representation."""
        num_checkpoints = len(self.manifest.checkpoints) if self.manifest else 0
        status = "initialized" if self.initialized else "not initialized"
        return (
            f"RunManager(run={self.run_id}, dir={self.run_dir}, "
            f"checkpoints={num_checkpoints}, status={status})"
        )
