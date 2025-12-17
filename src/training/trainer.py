"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.
"""

import time
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import CardAbstraction, RankBasedBucketing
from src.abstraction.equity_bucketing import EquityBucketing
from src.solver.base import BaseSolver
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage, InMemoryStorage, Storage
from src.training.checkpoint import CheckpointManager
from src.training.metrics import MetricsTracker
from src.utils.config import Config


class Trainer:
    """
    Orchestrates MCCFR training.

    Manages solver initialization, training loop, checkpointing,
    metrics tracking, and progress reporting.
    """

    def __init__(self, config: Config, run_id: Optional[str] = None):
        """
        Initialize trainer from configuration.

        Args:
            config: Configuration object
            run_id: Optional run ID (for resuming or explicit naming)
        """
        self.config = config

        # Initialize checkpoint manager FIRST to get run-specific directory
        checkpoint_base_dir = Path(config.get("training.checkpoint_dir", "data/checkpoints"))
        self.checkpoint_manager = CheckpointManager(
            checkpoint_base_dir,
            config_name=config.get("system.config_name", "default"),
            run_id=run_id,
            config=config.to_dict(),  # Pass full config for metadata
        )

        # Build components (storage needs the run-specific checkpoint_dir)
        self.action_abstraction = self._build_action_abstraction()
        self.card_abstraction = self._build_card_abstraction()
        self.storage = self._build_storage()

        # Build solver
        self.solver = self._build_solver()

        # Initialize metrics tracker
        self.metrics = MetricsTracker(window_size=config.get("training.log_frequency", 100))

    def _build_action_abstraction(self) -> ActionAbstraction:
        """Build action abstraction from config."""
        action_config = self.config.get_section("action_abstraction")
        return ActionAbstraction(action_config)

    def _build_card_abstraction(self) -> CardAbstraction:
        """Build card abstraction from config."""
        card_config = self.config.get_section("card_abstraction")
        abstraction_type = card_config.get("type", "rank_based")

        if abstraction_type == "rank_based":
            return RankBasedBucketing()
        elif abstraction_type == "equity_bucketing":
            # Load precomputed equity bucketing from file
            bucketing_path = card_config.get("bucketing_path")
            if not bucketing_path:
                raise ValueError("equity_bucketing type requires 'bucketing_path' in config")

            bucketing_path = Path(bucketing_path)
            if not bucketing_path.exists():
                raise FileNotFoundError(f"Equity bucketing file not found: {bucketing_path}")

            bucketing = EquityBucketing.load(bucketing_path)
            return bucketing
        else:
            raise ValueError(f"Unknown card abstraction type: {abstraction_type}")

    def _build_storage(self) -> Storage:
        """Build storage backend from config."""
        storage_config = self.config.get_section("storage")
        backend = storage_config.get("backend", "memory")

        if backend == "memory":
            return InMemoryStorage()
        elif backend == "disk":
            # Use the run-specific checkpoint directory from CheckpointManager
            checkpoint_dir = self.checkpoint_manager.checkpoint_dir
            cache_size = storage_config.get("cache_size", 100000)
            flush_frequency = storage_config.get("flush_frequency", 1000)

            return DiskBackedStorage(
                checkpoint_dir=checkpoint_dir,
                cache_size=cache_size,
                flush_frequency=flush_frequency,
            )
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    def _build_solver(self) -> BaseSolver:
        """Build solver from config."""
        solver_config = self.config.get_section("solver")
        game_config = self.config.get_section("game")
        system_config = self.config.get_section("system")

        # Merge configs for solver
        merged_config = {**game_config, **system_config}

        solver_type = solver_config.get("type", "mccfr")

        if solver_type == "mccfr":
            return MCCFRSolver(
                action_abstraction=self.action_abstraction,
                card_abstraction=self.card_abstraction,
                storage=self.storage,
                config=merged_config,
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    def train(
        self,
        num_iterations: Optional[int] = None,
        resume: bool = False,
    ) -> Dict:
        """
        Run training loop.

        Args:
            num_iterations: Number of iterations (overrides config if provided)
            resume: Whether to resume from latest checkpoint

        Returns:
            Training results dictionary
        """
        # Get iteration count
        if num_iterations is None:
            num_iterations = self.config.get("training.num_iterations", 1000)

        checkpoint_freq = self.config.get("training.checkpoint_frequency", 100)
        log_freq = self.config.get("training.log_frequency", 10)
        verbose = self.config.get("training.verbose", True)

        # Resume from checkpoint if requested
        start_iteration = 0
        if resume:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                start_iteration = latest_checkpoint["iteration"]
                print(f"Resuming from iteration {start_iteration}")
                # Note: Storage should already be loaded if using disk backend

        # Track training time
        training_start_time = time.time()

        # Training loop
        if verbose:
            iterator = tqdm(
                range(start_iteration, num_iterations),
                desc="Training",
                unit="iter",
            )
        else:
            iterator = range(start_iteration, num_iterations)

        try:
            for i in iterator:
                # Run one iteration
                utility = self.solver.train_iteration()

                # Log metrics
                self.metrics.log_iteration(
                    iteration=i + 1,
                    utility=utility,
                    num_infosets=self.solver.num_infosets(),
                )

                # Periodic logging
                if (i + 1) % log_freq == 0 and verbose:
                    summary = self.metrics.get_summary()
                    if isinstance(iterator, tqdm):
                        iterator.set_postfix(
                            {
                                "util": f"{summary['avg_utility']:+.2f}",
                                "infosets": f"{summary['avg_infosets']:.0f}",
                            }
                        )

                # Periodic checkpointing
                if (i + 1) % checkpoint_freq == 0:
                    elapsed_time = time.time() - training_start_time

                    # Get metrics for this checkpoint
                    summary = self.metrics.get_summary()
                    checkpoint_metrics = {
                        "avg_utility_p0": summary.get("avg_utility", 0.0),
                        "avg_utility_p1": -summary.get("avg_utility", 0.0),
                        "avg_infosets": summary.get("avg_infosets", 0),
                    }

                    # Save checkpoint with metrics
                    self.checkpoint_manager.save(
                        self.solver,
                        i + 1,
                        metrics=checkpoint_metrics,
                    )

                    # Update run statistics
                    self.checkpoint_manager.update_stats(
                        total_iterations=i + 1,
                        total_runtime_seconds=elapsed_time,
                        num_infosets=self.solver.num_infosets(),
                    )

            # Final checkpoint
            elapsed_time = time.time() - training_start_time
            summary = self.metrics.get_summary()
            final_metrics = {
                "avg_utility_p0": summary.get("avg_utility", 0.0),
                "avg_utility_p1": -summary.get("avg_utility", 0.0),
                "avg_infosets": summary.get("avg_infosets", 0),
            }

            self.checkpoint_manager.save(
                self.solver,
                num_iterations,
                metrics=final_metrics,
                tags=["final"],
            )

            # Update final stats
            self.checkpoint_manager.update_stats(
                total_iterations=num_iterations,
                total_runtime_seconds=elapsed_time,
                num_infosets=self.solver.num_infosets(),
            )

            # Mark run as completed
            self.checkpoint_manager.mark_completed()

            # Print final summary
            if verbose:
                self.metrics.print_summary()

            return {
                "total_iterations": num_iterations,
                "final_infosets": self.solver.num_infosets(),
                "avg_utility": self.metrics.get_avg_utility(),
                "elapsed_time": elapsed_time,
            }

        except Exception:
            # Mark run as failed on exception
            self.checkpoint_manager.mark_failed()
            raise

    def evaluate(self) -> Dict:
        """
        Evaluate current solver.

        Returns:
            Evaluation metrics
        """
        # TODO: Implement evaluation (head-to-head, exploitability)
        return {
            "num_infosets": self.solver.num_infosets(),
        }

    def __str__(self) -> str:
        """String representation."""
        return f"Trainer(solver={self.solver}, " f"config={self.config})"
