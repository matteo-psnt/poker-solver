#!/usr/bin/env python3
"""
Unified CLI for Poker Solver.

Provides a TUI for training, evaluation, precomputation, and run management.
"""

import signal
import sys
from pathlib import Path
from typing import Optional

import questionary
from questionary import Style

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.chart_handler import handle_view_preflop_chart
from src.cli.config_handler import select_config
from src.cli.precompute_handler import handle_precompute
from src.cli.training_handler import handle_resume, handle_train
from src.training.run.training_run import TrainingRun
from src.training.trainer import Trainer
from src.utils.config import Config

# Custom style
custom_style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:#6C6C6C"),
        ("instruction", ""),
        ("text", ""),
    ]
)


class SolverCLI:
    """Unified CLI for poker solver operations."""

    def __init__(self):
        """Initialize CLI."""
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.runs_dir = self.base_dir / "data" / "runs"
        self.equity_buckets_dir = self.base_dir / "data" / "equity_buckets"
        self.current_trainer: Optional[Trainer] = None

        # Setup Ctrl+C handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n[!] Interrupt received!")

        if self.current_trainer:
            print("Saving checkpoint before exit...")
            try:
                # Get current iteration from metrics
                summary = self.current_trainer.metrics.get_summary()
                current_iter = summary.get("total_iterations", 0)

                if current_iter > 0:
                    self.current_trainer.training_run.save_snapshot(
                        self.current_trainer.solver,
                        current_iter,
                        tags=["interrupted"],
                    )
                    self.current_trainer.training_run.update_stats(
                        total_iterations=current_iter,
                        total_runtime_seconds=self.current_trainer.metrics.get_elapsed_time(),
                        num_infosets=self.current_trainer.solver.num_infosets(),
                    )
                    print(f"[OK] Checkpoint saved at iteration {current_iter}")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint: {e}")

        print("Exiting...")
        sys.exit(0)

    def run(self):
        """Run the main TUI loop."""
        print("\n" + "=" * 60)
        print("POKER SOLVER CLI")
        print("=" * 60)

        while True:
            action = questionary.select(
                "\nWhat would you like to do?",
                choices=[
                    "Train Solver",
                    "Evaluate Solver",
                    "Precompute Equity Buckets",
                    "List Equity Buckets",
                    "View Past Runs",
                    "Resume Training",
                    "View Preflop Chart",
                    "Exit",
                ],
                style=custom_style,
            ).ask()

            if action is None or "Exit" in action:
                print("\nGoodbye!")
                break

            try:
                if "Train" in action and "Resume" not in action:
                    self.train_solver()
                elif "Evaluate" in action:
                    self.evaluate_solver()
                elif "Precompute" in action:
                    self.precompute_equity_buckets()
                elif "List Equity Buckets" in action:
                    from src.abstraction.equity.manager import EquityBucketManager

                    manager = EquityBucketManager()
                    manager.print_summary()
                    input("\nPress Enter to continue...")
                elif "View Past" in action:
                    self.view_runs()
                elif "Resume" in action:
                    self.resume_training()
                elif "Preflop Chart" in action:
                    self.view_preflop_chart()
            except KeyboardInterrupt:
                print("\n\n[!] Operation cancelled by user")
                continue
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback

                traceback.print_exc()
                input("\nPress Enter to continue...")

    def train_solver(self):
        """Train a new solver."""
        print("\nTrain Solver")
        print("=" * 60)

        config = select_config(self.config_dir, custom_style)
        if config is None:
            return

        try:
            trainer = handle_train(config, custom_style, self.runs_dir)
            self.current_trainer = trainer
        finally:
            self.current_trainer = None

        input("\nPress Enter to continue...")

    def evaluate_solver(self):
        """Evaluate a trained solver."""
        print("\nEvaluate Solver")
        print("=" * 60)

        # List available runs
        runs = TrainingRun.list_runs(self.runs_dir)

        if not runs:
            print("\n[ERROR] No trained runs found in data/runs/")
            input("Press Enter to continue...")
            return

        # Select run
        selected_run = questionary.select(
            "Select run to evaluate:",
            choices=runs + ["Cancel"],
            style=custom_style,
        ).ask()

        if selected_run == "Cancel" or selected_run is None:
            return

        # TODO: Implement evaluation
        print(f"\n[!] Evaluation not yet implemented for {selected_run}")
        print("   This will compare against baseline strategies")

        input("\nPress Enter to continue...")

    def precompute_equity_buckets(self):
        """Precompute equity buckets."""
        handle_precompute(custom_style)

    def view_runs(self):
        """View past training runs."""
        print("\nPast Training Runs")
        print("=" * 60)

        runs = TrainingRun.list_runs(self.runs_dir)

        if not runs:
            print("\n[ERROR] No training runs found")
            input("Press Enter to continue...")
            return

        # Select run to view
        selected = questionary.select(
            "Select run to view details:",
            choices=runs + ["Back"],
            style=custom_style,
        ).ask()

        if selected == "Back" or selected is None:
            return

        # Load and display run info
        training_run = TrainingRun.from_run_id(
            self.runs_dir,
            selected,
        )

        if training_run.run_metadata:
            meta = training_run.run_metadata
            print(f"\nRun: {selected}")
            print("-" * 60)
            print(f"Status: {meta.status}")
            print(f"Started: {meta.started_at}")
            if meta.completed_at:
                print(f"Completed: {meta.completed_at}")

            if meta.statistics:
                stats = meta.statistics
                print("\nStatistics:")
                print(f"  Iterations: {stats.total_iterations}")
                print(
                    f"  Runtime: {stats.total_runtime_seconds:.2f}s ({stats.total_runtime_seconds / 60:.1f}m)"
                )
                print(f"  Speed: {stats.iterations_per_second:.2f} it/s")
                print(f"  Infosets: {stats.num_infosets:,}")

        # Show checkpoints
        if training_run.manifest:
            snapshots = training_run.manifest.snapshots
            print(f"\nSnapshots: {len(snapshots)}")
            for snapshot in snapshots[:5]:  # Show first 5
                print(
                    f"  {snapshot.iteration:6d}: {snapshot.num_infosets:8,} infosets {snapshot.tags}"
                )
            if len(snapshots) > 5:
                print(f"  ... and {len(snapshots) - 5} more")

        input("\nPress Enter to continue...")

    def resume_training(self):
        """Resume training from a checkpoint."""
        print("\nResume Training")
        print("=" * 60)

        runs = TrainingRun.list_runs(self.runs_dir)

        if not runs:
            print("No training runs found.")
            input("Press Enter to continue...")
            return

        selected = questionary.select(
            "Select run to view:",
            choices=runs + ["Cancel"],
            style=custom_style,
        ).ask()

        if selected == "Cancel" or selected is None:
            return

        training_run = TrainingRun.from_run_id(self.runs_dir, selected)

        if not training_run.run_metadata or not training_run.run_metadata.config:
            print("\n[ERROR] No config found for this run")
            input("Press Enter to continue...")
            return

        config_dict = training_run.run_metadata.config
        config = Config.from_dict(config_dict)

        latest = training_run.get_latest_snapshot()
        if latest:
            print(f"\nLatest checkpoint: iteration {latest['iteration']}")
            print(f"Infosets: {latest['num_infosets']:,}")

        add_iters = questionary.text(
            "Additional iterations to run:",
            default="1000",
            style=custom_style,
        ).ask()

        if add_iters is None:
            return

        total_iters = latest["iteration"] + int(add_iters)
        config.set("training.num_iterations", total_iters)

        try:
            self.current_trainer = handle_resume(config, selected, latest["iteration"])
        finally:
            self.current_trainer = None

        input("\nPress Enter to continue...")

    def view_preflop_chart(self):
        """View preflop strategy chart from trained solver."""
        print("\nView Preflop Chart")
        print("=" * 60)

        handle_view_preflop_chart(self.runs_dir, self.base_dir, custom_style)


def main():
    """Main entry point."""
    cli = SolverCLI()
    cli.run()


if __name__ == "__main__":
    main()
