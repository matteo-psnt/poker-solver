"""
Training orchestration for MCCFR solver.

Manages the complete training loop: solver creation, iteration execution,
checkpointing, metrics tracking, and progress reporting.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import CardAbstraction
from src.abstraction.equity_bucketing import EquityBucketing
from src.solver.base import BaseSolver
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage, InMemoryStorage, Storage
from src.training.metrics import MetricsTracker
from src.training.training_run import TrainingRun
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

        # Initialize training run FIRST to get run-specific directory
        runs_base_dir = Path(config.get("training.runs_dir", "data/runs"))
        self.training_run = TrainingRun(
            base_dir=runs_base_dir,
            run_id=run_id,
            config_name=config.get("system.config_name", "default"),
            config=config.to_dict(),  # Pass full config for metadata
        )

        # Build components (storage needs the run-specific directory)
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
        game_config = self.config.get_section("game")
        big_blind = game_config.get("big_blind", 2)
        return ActionAbstraction(action_config, big_blind=big_blind)

    def _build_card_abstraction(self) -> CardAbstraction:
        """Build card abstraction from config."""
        card_config = self.config.get_section("card_abstraction")
        abstraction_type = card_config.get("type", "equity_bucketing")

        if abstraction_type == "equity_bucketing":
            # Check if using new config-based system
            abstraction_config = card_config.get("config")

            if abstraction_config:
                # NEW: Load by config name with auto-prompt
                from src.abstraction.abstraction_metadata import AbstractionManager

                manager = AbstractionManager()
                bucketing_path = manager.find_or_compute(
                    config_name=abstraction_config,
                    auto_compute=False,
                    prompt_user=True,  # Prompt user if not found
                )
                bucketing = EquityBucketing.load(bucketing_path)
                return bucketing

            # OLD: Direct file path (backward compatibility)
            bucketing_path = card_config.get("bucketing_path")
            if not bucketing_path:
                raise ValueError(
                    "equity_bucketing requires either 'config' or 'bucketing_path'.\n"
                    "Recommended: Use 'config: production' to reference an abstraction config."
                )

            bucketing_path = Path(bucketing_path)
            if not bucketing_path.exists():
                raise FileNotFoundError(
                    f"Equity bucketing file not found: {bucketing_path}\n"
                    "Please run 'Precompute Equity Buckets' from the CLI first."
                )

            bucketing = EquityBucketing.load(bucketing_path)
            return bucketing
        else:
            raise ValueError(
                f"Unknown card abstraction type: {abstraction_type}\n"
                "Only 'equity_bucketing' is supported."
            )

    def _build_storage(self) -> Storage:
        """Build storage backend from config."""
        storage_config = self.config.get_section("storage")
        backend = storage_config.get("backend", "memory")

        if backend == "memory":
            return InMemoryStorage()
        elif backend == "disk":
            # Use the run-specific directory from TrainingRun
            run_dir = self.training_run.run_dir
            cache_size = storage_config.get("cache_size", 100000)
            flush_frequency = storage_config.get("flush_frequency", 1000)

            return DiskBackedStorage(
                checkpoint_dir=run_dir,
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

        # Resume from snapshot if requested
        start_iteration = 0
        if resume:
            latest_snapshot = self.training_run.get_latest_snapshot()
            if latest_snapshot:
                start_iteration = latest_snapshot["iteration"]
                print(f"Resuming from iteration {start_iteration}")
                # Note: Storage should already be loaded if using disk backend

        # Track training time
        training_start_time = time.time()

        # Training loop
        if verbose:
            iterator: Union[range, tqdm] = tqdm(
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

                # Periodic snapshotting
                if (i + 1) % checkpoint_freq == 0:
                    elapsed_time = time.time() - training_start_time

                    # Get metrics for this snapshot
                    summary = self.metrics.get_summary()
                    snapshot_metrics = {
                        "avg_utility_p0": summary.get("avg_utility", 0.0),
                        "avg_utility_p1": -summary.get("avg_utility", 0.0),
                        "avg_infosets": summary.get("avg_infosets", 0),
                    }

                    # Save snapshot with metrics
                    self.training_run.save_snapshot(
                        self.solver,
                        i + 1,
                        metrics=snapshot_metrics,
                    )

                    # Update run statistics
                    self.training_run.update_stats(
                        total_iterations=i + 1,
                        total_runtime_seconds=elapsed_time,
                        num_infosets=self.solver.num_infosets(),
                    )

            # Final snapshot
            elapsed_time = time.time() - training_start_time
            summary = self.metrics.get_summary()
            final_metrics = {
                "avg_utility_p0": summary.get("avg_utility", 0.0),
                "avg_utility_p1": -summary.get("avg_utility", 0.0),
                "avg_infosets": summary.get("avg_infosets", 0),
            }

            self.training_run.save_snapshot(
                self.solver,
                num_iterations,
                metrics=final_metrics,
                tags=["final"],
            )

            # Update final stats
            self.training_run.update_stats(
                total_iterations=num_iterations,
                total_runtime_seconds=elapsed_time,
                num_infosets=self.solver.num_infosets(),
            )

            # Mark experiment as completed
            self.training_run.mark_completed()

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
            # Mark experiment as failed on exception
            self.training_run.mark_failed()
            raise

    def evaluate(
        self,
        num_samples: int = 10000,
        num_rollouts_per_infoset: int = 100,
    ) -> Dict:
        """
        Evaluate current solver using exploitability estimation.

        Args:
            num_samples: Number of game samples for exploitability estimation
            num_rollouts_per_infoset: Number of rollouts per infoset for BR approximation

        Returns:
            Evaluation metrics including exploitability estimates
        """
        from src.evaluation.exploitability import compute_exploitability
        from src.solver.mccfr import MCCFRSolver

        # Cast to MCCFRSolver for type checking
        if not isinstance(self.solver, MCCFRSolver):
            raise TypeError("Exploitability computation requires MCCFRSolver")

        results = compute_exploitability(
            self.solver,
            num_samples=num_samples,
            num_rollouts_per_infoset=num_rollouts_per_infoset,
        )

        return {
            "num_infosets": self.solver.num_infosets(),
            "exploitability_mbb": results["exploitability_mbb"],
            "std_error_mbb": results["std_error_mbb"],
            "confidence_95_mbb": results["confidence_95_mbb"],
        }

    def __str__(self) -> str:
        """String representation."""
        return f"Trainer(solver={self.solver}, config={self.config})"
