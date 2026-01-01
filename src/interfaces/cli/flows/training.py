"""Training and evaluation operations for CLI."""

import multiprocessing as mp
from pathlib import Path
from typing import cast

from src.interfaces.cli.flows.combo_precompute import handle_combo_precompute
from src.interfaces.cli.flows.config import select_config
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.training import services
from src.pipeline.training.components import (
    build_card_abstraction,
)
from src.pipeline.training.services import EvaluationOutput
from src.shared.config import Config


def train_solver(ctx: CliContext) -> None:
    """Train a new solver."""
    ui.header("Train Solver")

    config = select_config(ctx)
    if config is None:
        return

    # Keep CLI run-related flows aligned with the selected training configuration.
    ctx.set_runs_dir(config.training.runs_dir)

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

    runs = services.list_runs(ctx.runs_dir)

    if not runs:
        ui.error(f"No trained runs found in {ctx.runs_dir}")
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

    print("\nLoading solver and running exploitability estimate...")
    try:
        output = services.evaluate_run(
            run_dir=run_dir,
            num_samples=num_samples,
            num_rollouts=num_rollouts,
            use_average_strategy=use_average,
            seed=seed,
        )
    except FileNotFoundError as e:
        ui.error(str(e))
        print(
            "\nThe combo abstraction used by this run is missing. "
            "It may have been deleted or moved."
        )
        ui.pause()
        return
    except ValueError as e:
        ui.error(str(e))
        ui.pause()
        return

    _print_evaluation_results(output)
    ui.pause()


def _print_evaluation_results(output: EvaluationOutput) -> None:
    print(f"  Loaded {output.infosets:,} infosets")
    print()
    results = output.results

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


def view_runs(ctx: CliContext) -> None:
    """View past training runs."""
    ui.header("Past Training Runs")

    runs = services.list_runs(ctx.runs_dir)

    if not runs:
        ui.error(f"No training runs found in {ctx.runs_dir}")
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
    meta = services.load_run_metadata(run_dir)

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

    runs = services.list_runs(ctx.runs_dir)

    if not runs:
        ui.error(f"No training runs found in {ctx.runs_dir}")
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
    meta = services.load_run_metadata(run_dir)

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

    _resume_training(run_dir, additional_iters)
    ui.pause()


def _prompt_num_workers(ctx: CliContext) -> int | None:
    default_workers = mp.cpu_count()

    return prompts.prompt_int(
        ctx,
        f"Number of workers (default: {default_workers}):",
        default=default_workers,
        min_value=1,
    )


def _run_precompute_and_verify(ctx: CliContext, config: Config) -> bool:
    """Run precomputation and verify it completed successfully."""
    print("\n" + "=" * 60)
    print("RUNNING PRECOMPUTATION")
    print("=" * 60)
    handle_combo_precompute(ctx)

    try:
        build_card_abstraction(config)
        print("\n✓ Precomputation completed successfully!")
        print("Continuing with training setup...\n")
        return True
    except (FileNotFoundError, ValueError):
        ui.error("\nPrecomputation did not complete successfully.")
        print("Training cancelled.")
        return False


def _ensure_combo_abstraction(ctx: CliContext, config: Config) -> bool:
    """
    Ensure combo abstraction exists for the given config.

    If the abstraction is missing, prompts the user to run precomputation.

    Returns:
        True if abstraction exists or was successfully created, False otherwise
    """
    try:
        build_card_abstraction(config)
        return True
    except FileNotFoundError as e:
        ui.error(str(e))
        print()

        run_precompute = prompts.confirm(
            ctx,
            "Would you like to run precomputation now?",
            default=True,
        )

        if not run_precompute:
            print("\nTraining cancelled. Please run precomputation from the main menu first.")
            return False

        return _run_precompute_and_verify(ctx, config)

    except ValueError as e:
        ui.error(str(e))
        print()

        if "hash mismatch" not in str(e).lower():
            return False

        run_precompute = prompts.confirm(
            ctx,
            "Would you like to recompute the abstraction now?",
            default=True,
        )

        if not run_precompute:
            print("\nTraining cancelled.")
            return False

        return _run_precompute_and_verify(ctx, config)


def _start_training(config: Config, num_workers: int) -> None:
    trainer = services.create_training_session(config)

    print("\nStarting training...")
    print(f"Run directory: {trainer.run_dir}")
    print(f"Checkpoint frequency: every {config.training.checkpoint_frequency} iterations")
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")
    services.run_training(trainer, num_workers=num_workers)


def _resume_training(run_dir: Path, additional_iters: int) -> None:
    trainer, latest_iter = services.create_resumed_session(run_dir)

    print(f"\nResuming training from iteration {latest_iter}...")
    target_total = latest_iter + additional_iters
    print(f"Target: {target_total} iterations (+{additional_iters})")
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")
    services.run_training(trainer, num_iterations=additional_iters)
