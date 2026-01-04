"""Training operations for CLI."""

import multiprocessing as mp
from pathlib import Path
from typing import Optional

import questionary

from src.training.trainer import TrainingSession
from src.utils.config import Config


def handle_train(
    config: Config,
    custom_style,
    checkpoint_dir: Path,
) -> Optional[TrainingSession]:
    """
    Handle training a new solver.

    Args:
        config: Training configuration
        custom_style: Questionary style
        checkpoint_dir: Directory for checkpoints

    Returns:
        TrainingSession instance or None if cancelled
    """
    # Check combo abstraction requirement
    abstraction_path = config.card_abstraction.abstraction_path
    abstraction_config = config.card_abstraction.config

    if abstraction_path:
        if not Path(abstraction_path).exists():
            print(f"\n[ERROR] Combo abstraction file not found: {abstraction_path}")
            print("   Please precompute combo abstraction first (from main menu)")
            return None
    elif abstraction_config:
        # Check if any abstraction exists
        base_path = Path("data/combo_abstraction")
        abstraction_found = False
        if base_path.exists():
            for path in base_path.iterdir():
                if path.is_dir() and (path / "combo_abstraction.pkl").exists():
                    abstraction_found = True
                    break
        if not abstraction_found:
            print("\n[ERROR] No combo abstraction found.")
            print("   Please precompute combo abstraction first (from main menu)")
            return None

    # Ask for number of workers (training is always parallel)
    default_workers = mp.cpu_count()

    num_workers = questionary.text(
        f"Number of workers (default: {default_workers}):",
        default=str(default_workers),
        style=custom_style,
    ).ask()

    if num_workers is None:
        return None

    try:
        num_workers = int(num_workers)
        if num_workers < 1:
            print("[ERROR] Number of workers must be at least 1")
            input("Press Enter to continue...")
            return None
    except ValueError:
        print("[ERROR] Invalid number")
        input("Press Enter to continue...")
        return None

    # Create trainer
    trainer = TrainingSession(config)

    # Start training
    print("\nStarting training...")
    print(f"Run directory: {trainer.run_dir}")
    print(f"Checkpoint frequency: every {config.training.checkpoint_frequency} iterations")
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

    trainer.train(num_workers=num_workers)

    return trainer


def handle_resume(config: Config, run_id: str, latest_iter: int) -> TrainingSession:
    """
    Resume training from checkpoint.

    Args:
        config: Configuration
        run_id: Run identifier
        latest_iter: Latest iteration number

    Returns:
        TrainingSession instance
    """
    runs_dir = Path(config.training.runs_dir)
    run_dir = runs_dir / run_id
    trainer = TrainingSession.resume(run_dir)

    print(f"\nResuming training from iteration {latest_iter}...")
    print(
        f"Target: {config.training.num_iterations} iterations (+{config.training.num_iterations - latest_iter})"
    )
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

    trainer.train()

    return trainer
