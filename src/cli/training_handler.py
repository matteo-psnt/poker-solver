"""Training operations for CLI."""

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Union

import questionary

from src.training.trainer import Trainer
from src.utils.config import Config

if TYPE_CHECKING:
    from src.training.parallel_trainer import ParallelTrainer


def handle_train(
    config: Config,
    custom_style,
    checkpoint_dir: Path,
) -> Union[Trainer, "ParallelTrainer", None]:
    """
    Handle training a new solver.

    Args:
        config: Training configuration
        custom_style: Questionary style
        checkpoint_dir: Directory for checkpoints

    Returns:
        Trainer instance or None if cancelled
    """
    # Check equity bucketing requirement
    if config.get("card_abstraction.type") == "equity_bucketing":
        bucketing_path = config.get("card_abstraction.bucketing_path")
        if bucketing_path and not Path(bucketing_path).exists():
            print(f"\n[ERROR] Equity bucketing file required but not found: {bucketing_path}")
            print("   Please precompute equity buckets first (option 3 from main menu)")
            return None

    # Ask about parallel training
    use_parallel = questionary.confirm(
        "Use parallel training? (faster with multiple CPU cores)",
        default=True,
        style=custom_style,
    ).ask()

    if use_parallel is None:
        return None

    num_workers = None
    if use_parallel:
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
    print("\nInitializing trainer...")
    if use_parallel:
        from src.training.parallel_trainer import ParallelTrainer

        trainer: Union[Trainer, "ParallelTrainer"] = ParallelTrainer(
            config, num_workers=num_workers
        )
    else:
        trainer = Trainer(config)

    # Start training
    print(f"\nStarting training for {config.get('training.num_iterations')} iterations...")
    if hasattr(trainer, "run_manager"):
        print(f"Run directory: {trainer.run_manager.run_dir}")
    print(f"Checkpoint frequency: every {config.get('training.checkpoint_frequency')} iterations")
    if use_parallel:
        print(f"Parallel workers: {num_workers}")
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

    results = trainer.train()

    print("\n[OK] Training completed!")
    print(f"   Total iterations: {results.get('iterations', results.get('total_iterations'))}")
    print(f"   Final infosets: {results.get('num_infosets', results.get('final_infosets'))}")
    print(
        f"   Average utility: {results.get('metrics', {}).get('avg_utility', results.get('avg_utility', 0)):.4f}"
    )
    print(f"   Elapsed time: {results.get('elapsed_time', 0):.2f}s")

    return trainer


def handle_resume(config: Config, run_id: str, latest_iter: int) -> Trainer:
    """
    Resume training from checkpoint.

    Args:
        config: Training configuration
        run_id: Run ID to resume
        latest_iter: Latest iteration number

    Returns:
        Trainer instance
    """
    print("\nResuming trainer...")
    trainer = Trainer(config, run_id=run_id)

    print(f"\nResuming training from iteration {latest_iter}...")
    print(
        f"Target: {config.get('training.num_iterations')} iterations (+{config.get('training.num_iterations') - latest_iter})"
    )
    print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

    results = trainer.train(resume=True)

    print("\n[OK] Training completed!")
    print(f"   Total iterations: {results['total_iterations']}")
    print(f"   Final infosets: {results['final_infosets']}")

    return trainer
