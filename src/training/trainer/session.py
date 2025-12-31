"""Training session class for MCCFR solver orchestration."""

import concurrent.futures
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any

from src.solver.storage.helpers import get_missing_checkpoint_files
from src.training import components
from src.training.metrics import MetricsTracker
from src.training.run_tracker import RunTracker
from src.utils.config import Config

from . import checkpointing, partitioned


class TrainingSession:
    """Orchestrates MCCFR training end to end."""

    def __init__(
        self,
        config: Config,
        run_id: str | None = None,
        run_tracker: RunTracker | None = None,
    ):
        self.config = config

        self.run_tracker = run_tracker
        if self.run_tracker:
            self.run_dir = self.run_tracker.run_dir
        else:
            runs_base_dir = Path(self.config.training.runs_dir)
            if run_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"run-{timestamp}"
            self.run_dir = runs_base_dir / run_id

        self._checkpoint_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending_checkpoint: concurrent.futures.Future[float] | None = None

        try:
            self.action_abstraction = components.build_action_abstraction(config)
            self.card_abstraction = components.build_card_abstraction(config)
            action_config_hash = self.action_abstraction.get_config_hash()

            if self.run_tracker is None:
                self.run_tracker = RunTracker(
                    run_dir=self.run_dir,
                    config_name=self.config.system.config_name,
                    config=config,
                    action_config_hash=action_config_hash,
                )
            else:
                self.run_tracker.verify_action_config_hash(action_config_hash)

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.storage = components.build_storage(
                config,
                run_dir=self.run_dir,
                run_metadata=self.run_tracker.metadata,
            )
            self.solver = components.build_solver(
                config, self.action_abstraction, self.card_abstraction, self.storage
            )

            self.metrics = MetricsTracker()

            if self.config.storage.checkpoint_enabled:
                self._checkpoint_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="checkpoint"
                )
        except Exception:
            if self.run_tracker is not None:
                self.run_tracker.mark_failed(cleanup_if_empty=True)
            raise

    @classmethod
    def resume(cls, run_dir: str | Path, checkpoint_id: int | None = None) -> "TrainingSession":
        """Resume training from a checkpoint in an existing run directory."""
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")

        run_tracker = RunTracker.load(run_path)
        metadata = run_tracker.metadata

        if checkpoint_id is not None:
            checkpoint_iter = checkpoint_id
        else:
            checkpoint_iter = metadata.iterations if metadata.iterations > 0 else None

        if checkpoint_iter is None:
            raise FileNotFoundError(f"No checkpoint found in {run_path}")

        missing_files = get_missing_checkpoint_files(run_path)
        if missing_files:
            raise ValueError(f"Checkpoint is incomplete. Missing files: {missing_files}")

        if metadata.config is None:
            raise ValueError(f"Missing config in run metadata: {run_tracker.metadata_path}")
        config = metadata.config

        session = cls(config, run_id=run_path.name, run_tracker=run_tracker)

        if session.storage.num_infosets() == 0:
            raise ValueError(
                "Failed to load checkpoint data. Storage has 0 infosets.\n"
                f"Expected to load from: {run_path}"
            )

        session.solver.iteration = checkpoint_iter

        assert session.run_tracker is not None
        session.run_tracker.mark_resumed()

        print(f"✅ Resumed from checkpoint at iteration {checkpoint_iter}")

        return session

    def __del__(self):
        """Cleanup on deletion."""
        checkpointing.shutdown_checkpoint_executor(self)

    @property
    def verbose(self) -> bool:
        """Get verbose setting from config."""
        return self.config.training.verbose

    def train(
        self,
        num_iterations: int | None = None,
        num_workers: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Run parallel training using hash-partitioned infosets."""
        if num_iterations is None:
            num_iterations = self.config.training.num_iterations

        if num_workers is None:
            num_workers = mp.cpu_count()

        return partitioned.train_partitioned(self, num_iterations, num_workers, batch_size)

    def evaluate(
        self,
        num_samples: int = 10000,
        num_rollouts_per_infoset: int = 100,
        use_average_strategy: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate current solver using exploitability estimation."""
        results = components.evaluate_solver_exploitability(
            self.solver,
            num_samples=num_samples,
            use_average_strategy=use_average_strategy,
            num_rollouts_per_infoset=num_rollouts_per_infoset,
            seed=seed,
        )

        return {
            "num_infosets": self.solver.num_infosets(),
            "exploitability_mbb": results["exploitability_mbb"],
            "std_error_mbb": results["std_error_mbb"],
            "confidence_95_mbb": results["confidence_95_mbb"],
        }

    def __str__(self) -> str:
        """String representation."""
        return f"TrainingSession(solver={self.solver}, config={self.config})"
