"""Training and evaluation operations for CLI."""

import multiprocessing as mp
from pathlib import Path
from typing import cast

from src.cli.flows.config import select_config
from src.cli.ui import prompts, ui
from src.cli.ui.context import CliContext
from src.evaluation.exploitability import compute_exploitability
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.in_memory import InMemoryStorage
from src.training.components import (
    build_action_abstraction,
    build_card_abstraction,
    build_solver,
)
from src.training.run_tracker import RunTracker
from src.training.trainer import TrainingSession
from src.utils.config import Config


def train_solver(ctx: CliContext) -> None:
    """Train a new solver."""
    ui.header("Train Solver")

    config = select_config(ctx)
    if config is None:
        return

    if not _ensure_combo_abstraction(ctx, config):
        ui.pause()
        return

    num_workers = _prompt_num_workers(ctx)
    if num_workers is None:
        return

    _start_training(config, num_workers)
    ui.pause()


def evaluate_solver(ctx: CliContext) -> None:
    """Evaluate a trained solver."""
    ui.header("Evaluate Solver")

    runs = RunTracker.list_runs(ctx.runs_dir)

    if not runs:
        ui.error("No trained runs found in data/runs/")
        ui.pause()
        return

    selected_run = prompts.select(
        ctx,
        "Select run to evaluate:",
        choices=runs + ["Cancel"],
    )

    if selected_run is None or selected_run == "Cancel":
        return

    run_dir = ctx.runs_dir / selected_run
    tracker = RunTracker.load(run_dir)
    meta = tracker.metadata

    config = meta.config

    num_samples = prompts.prompt_int(
        ctx,
        "Num samples:",
        default=500,
        min_value=1,
    )
    if num_samples is None:
        return

    num_rollouts = prompts.prompt_int(
        ctx,
        "Rollouts per infoset:",
        default=50,
        min_value=1,
    )
    if num_rollouts is None:
        return

    strategy_choice = prompts.select(
        ctx,
        "Strategy to evaluate:",
        choices=["Average (recommended)", "Current"],
    )
    if strategy_choice is None:
        return

    def _validate_seed(value: str) -> bool | str:
        if not value.strip():
            return True
        if value.isdigit():
            return True
        return "Enter a whole number"

    seed_text = prompts.text(
        ctx,
        "Random seed (blank for random):",
        default="",
        validate=_validate_seed,
    )
    if seed_text is None:
        return

    use_average = strategy_choice.startswith("Average")
    seed = int(seed_text) if seed_text.strip() else None

    print("\nLoading solver...")
    storage = InMemoryStorage(checkpoint_dir=run_dir)
    action_abstraction = build_action_abstraction(config)
    card_abstraction = build_card_abstraction(
        config,
        prompt_user=False,
        auto_compute=False,
    )
    solver = build_solver(config, action_abstraction, card_abstraction, storage)

    print(f"  Loaded {storage.num_infosets():,} infosets")
    print("\nRunning exploitability estimate...\n")
    assert isinstance(solver, MCCFRSolver), f"Expected MCCFRSolver, got {type(solver)}"
    results = compute_exploitability(
        solver,
        num_samples=num_samples,
        use_average_strategy=use_average,
        num_rollouts_per_infoset=num_rollouts,
        seed=seed,
    )

    print("Exploitability Results")
    print("-" * 60)
    print(f"Exploitability: {results['exploitability_mbb']:.2f} mbb/g")
    print(f"Std Error:      {results['std_error_mbb']:.2f} mbb/g")
    confidence_interval = cast(tuple[float, float], results["confidence_95_mbb"])
    ci_lower, ci_upper = confidence_interval[0], confidence_interval[1]
    print(f"95% CI:         [{ci_lower:.2f}, {ci_upper:.2f}] mbb/g")
    print(f"BR Utility P0:  {results['player_0_br_utility']:.4f}")
    print(f"BR Utility P1:  {results['player_1_br_utility']:.4f}")
    print(f"Samples:        {results['num_samples']}")
    print(
        "\nRule of thumb (mbb/g): 100+ very exploitable, 20-100 weak, 5-20 decent, "
        "1-5 good, 0.1-1 strong, <0.1 near-optimal."
    )

    ui.pause()


def view_runs(ctx: CliContext) -> None:
    """View past training runs."""
    ui.header("Past Training Runs")

    runs = RunTracker.list_runs(ctx.runs_dir)

    if not runs:
        ui.error("No training runs found")
        ui.pause()
        return

    selected = prompts.select(
        ctx,
        "Select run to view details:",
        choices=runs + ["Back"],
    )

    if selected is None or selected == "Back":
        return

    run_dir = ctx.runs_dir / selected
    tracker = RunTracker.load(run_dir)
    meta = tracker.metadata

    print(f"\nRun: {selected}")
    print("-" * 60)
    print(f"Status: {meta.status or 'unknown'}")
    print(f"Started: {meta.started_at or 'N/A'}")
    if meta.completed_at:
        print(f"Completed: {meta.completed_at}")

    print("\nStatistics:")
    print(f"  Iterations: {meta.iterations}")
    runtime = meta.runtime_seconds
    print(f"  Runtime: {runtime:.2f}s ({runtime / 60:.1f}m)")
    if runtime > 0 and meta.iterations > 0:
        print(f"  Speed: {meta.iterations / runtime:.2f} it/s")
    print(f"  Infosets: {meta.num_infosets:,}")

    print(f"\nConfig: {meta.config_name or 'unknown'}")

    ui.pause()


def resume_training(ctx: CliContext) -> None:
    """Resume training from a checkpoint."""
    ui.header("Resume Training")

    runs = RunTracker.list_runs(ctx.runs_dir)

    if not runs:
        ui.error("No training runs found.")
        ui.pause()
        return

    selected = prompts.select(
        ctx,
        "Select run to view:",
        choices=runs + ["Cancel"],
    )

    if selected is None or selected == "Cancel":
        return

    run_dir = ctx.runs_dir / selected
    tracker = RunTracker.load(run_dir)
    meta = tracker.metadata

    latest_iter = meta.iterations
    if latest_iter > 0:
        print(f"\nLatest checkpoint: iteration {latest_iter}")
        print(f"Infosets: {meta.num_infosets:,}")

    additional_iters = prompts.prompt_int(
        ctx,
        "Additional iterations to run:",
        default=1000,
        min_value=1,
    )

    if additional_iters is None:
        return

    _resume_training(run_dir, latest_iter, additional_iters)
    ui.pause()


def _prompt_num_workers(ctx: CliContext) -> int | None:
    default_workers = mp.cpu_count()

    return prompts.prompt_int(
        ctx,
        f"Number of workers (default: {default_workers}):",
        default=default_workers,
        min_value=1,
    )


def _ensure_combo_abstraction(ctx: CliContext, config: Config) -> bool:
    abstraction_path = config.card_abstraction.abstraction_path
    abstraction_config = config.card_abstraction.config

    if abstraction_path:
        if not Path(abstraction_path).exists():
            ui.error(f"Combo abstraction file not found: {abstraction_path}")
            print("   Please precompute combo abstraction first (from main menu)")
            return False
    elif abstraction_config:
        base_path = ctx.base_dir / "data" / "combo_abstraction"
        abstraction_found = False
        if base_path.exists():
            for path in base_path.iterdir():
                if path.is_dir() and (path / "combo_abstraction.pkl").exists():
                    abstraction_found = True
                    break
        if not abstraction_found:
            ui.error("No combo abstraction found.")
            print("   Please precompute combo abstraction first (from main menu)")
            return False

    return True


def _start_training(config: Config, num_workers: int) -> TrainingSession:
    trainer = TrainingSession(config)

    print("\nStarting training...")
    print(f"Run directory: {trainer.run_dir}")
    print(f"Checkpoint frequency: every {config.training.checkpoint_frequency} iterations")
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

    trainer.train(num_workers=num_workers)

    return trainer


def _resume_training(run_dir: Path, latest_iter: int, additional_iters: int) -> TrainingSession:
    trainer = TrainingSession.resume(run_dir)

    print(f"\nResuming training from iteration {latest_iter}...")
    target_total = latest_iter + additional_iters
    print(f"Target: {target_total} iterations (+{additional_iters})")
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

    trainer.train(num_iterations=additional_iters)

    return trainer
