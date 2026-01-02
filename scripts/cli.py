#!/usr/bin/env python3
"""
Unified CLI for Poker Solver.

Provides a TUI for training, evaluation, precomputation, and run management.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional

import questionary
from questionary import Style

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.chart_handler import handle_view_preflop_chart
from src.cli.combo_handler import (
    handle_combo_analyze_bucketing,
    handle_combo_coverage,
    handle_combo_info,
    handle_combo_precompute,
    handle_combo_test_lookup,
)
from src.cli.config_handler import select_config
from src.cli.training_handler import handle_resume, handle_train
from src.training.run_tracker import RunTracker
from src.training.trainer import TrainingSession
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
        self.current_trainer: Optional[TrainingSession] = None

        # Use default Ctrl+C behavior; training handles interrupts internally.

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
                    "Combo Abstraction Tools",
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

            # Skip separator lines
            if action.startswith("---"):
                continue

            try:
                if "Train" in action and "Resume" not in action:
                    self.train_solver()
                elif "Evaluate" in action:
                    self.evaluate_solver()
                elif "Combo Abstraction" in action:
                    self.combo_abstraction_menu()
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
        runs = RunTracker.list_runs(self.runs_dir)

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

    def view_runs(self):
        """View past training runs."""
        print("\nPast Training Runs")
        print("=" * 60)

        runs = RunTracker.list_runs(self.runs_dir)

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
        run_dir = self.runs_dir / selected
        tracker = RunTracker.load(run_dir)
        meta = tracker.metadata

        print(f"\nRun: {selected}")
        print("-" * 60)
        print(f"Status: {meta.get('status', 'unknown')}")
        print(f"Started: {meta.get('started_at', 'N/A')}")
        if meta.get("completed_at"):
            print(f"Completed: {meta['completed_at']}")

        print("\nStatistics:")
        print(f"  Iterations: {meta.get('iterations', 0)}")
        runtime = meta.get("runtime_seconds", 0)
        print(f"  Runtime: {runtime:.2f}s ({runtime / 60:.1f}m)")
        if runtime > 0 and meta.get("iterations", 0) > 0:
            print(f"  Speed: {meta['iterations'] / runtime:.2f} it/s")
        print(f"  Infosets: {meta.get('num_infosets', 0):,}")

        # Show config
        print(f"\nConfig: {meta.get('config_name', 'unknown')}")

        input("\nPress Enter to continue...")

    def resume_training(self):
        """Resume training from a checkpoint."""
        print("\nResume Training")
        print("=" * 60)

        runs = RunTracker.list_runs(self.runs_dir)

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

        run_dir = self.runs_dir / selected
        tracker = RunTracker.load(run_dir)
        meta = tracker.metadata

        if not meta.get("config"):
            print("\n[ERROR] No config found for this run")
            input("Press Enter to continue...")
            return

        config_dict = meta["config"]
        config = Config.from_dict(config_dict)

        latest_iter = meta.get("iterations", 0)
        if latest_iter > 0:
            print(f"\nLatest checkpoint: iteration {latest_iter}")
            print(f"Infosets: {meta.get('num_infosets', 0):,}")

        add_iters = questionary.text(
            "Additional iterations to run:",
            default="1000",
            style=custom_style,
        ).ask()

        if add_iters is None:
            return

        total_iters = latest_iter + int(add_iters)
        config = config.merge({"training": {"num_iterations": total_iters}})

        try:
            self.current_trainer = handle_resume(config, selected, latest_iter)
        finally:
            self.current_trainer = None

        input("\nPress Enter to continue...")

    def combo_abstraction_menu(self):
        """Show combo abstraction tools submenu."""
        while True:
            action = questionary.select(
                "\nCombo Abstraction Tools:",
                choices=[
                    "Precompute Abstraction",
                    "View Abstraction Info",
                    "Test Bucket Lookup",
                    "Analyze Bucketing Patterns",
                    "Analyze Coverage (Fallback Rate)",
                    "Back",
                ],
                style=custom_style,
            ).ask()

            if action is None or action == "Back":
                return

            try:
                if "Precompute" in action:
                    handle_combo_precompute()
                    input("\nPress Enter to continue...")
                elif "View" in action:
                    handle_combo_info()
                    input("\nPress Enter to continue...")
                elif "Test" in action:
                    handle_combo_test_lookup()
                    input("\nPress Enter to continue...")
                elif "Analyze Bucketing" in action:
                    handle_combo_analyze_bucketing()
                    input("\nPress Enter to continue...")
                elif "Analyze Coverage" in action:
                    handle_combo_coverage()
                    input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\n[!] Operation cancelled by user")
                continue
            except Exception as e:
                print(f"\n[ERROR] {e}")
                traceback.print_exc()
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
