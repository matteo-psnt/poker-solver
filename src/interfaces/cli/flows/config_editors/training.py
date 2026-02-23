"""Training section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import try_merge
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_training_params(ctx: CliContext, config: Config) -> Config:
    print("Training Parameters")
    print("-" * 40)
    ui.info("Total iterations run x iterations_per_worker = total work per training session.")

    iterations = prompts.prompt_int(
        ctx,
        "Number of iterations:",
        default=config.training.num_iterations,
        min_value=1,
    )
    if iterations is None:
        return config

    checkpoint_freq = prompts.prompt_int(
        ctx,
        "Checkpoint frequency (save every N iterations):",
        default=config.training.checkpoint_frequency,
        min_value=1,
        max_value=iterations,
    )
    if checkpoint_freq is None:
        return config

    iterations_per_worker = prompts.prompt_int(
        ctx,
        "Iterations per worker (batch size for parallel training):",
        default=config.training.iterations_per_worker,
        min_value=1,
    )
    if iterations_per_worker is None:
        return config

    verbose = prompts.confirm(ctx, "Verbose output?", default=config.training.verbose)
    if verbose is None:
        return config

    runs_dir = prompts.text(
        ctx,
        "Output directory for training runs:",
        default=config.training.runs_dir,
    )
    if runs_dir is None:
        return config

    return try_merge(
        config,
        {
            "training": {
                "num_iterations": iterations,
                "checkpoint_frequency": checkpoint_freq,
                "iterations_per_worker": iterations_per_worker,
                "verbose": verbose,
                "runs_dir": runs_dir,
            }
        },
    )
