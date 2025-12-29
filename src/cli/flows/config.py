"""Configuration handling for CLI."""

from typing import Optional

from src.cli.ui import prompts, ui
from src.cli.ui.context import CliContext
from src.utils.config import Config
from src.utils.config_loader import load_config


def select_config(ctx: CliContext) -> Optional[Config]:
    """
    Select and optionally edit a config file.

    Args:
        ctx: CLI context

    Returns:
        Loaded Config object or None if cancelled
    """
    training_config_dir = ctx.config_dir / "training"
    config_files = sorted(training_config_dir.glob("*.yaml"))

    if not config_files:
        ui.error(f"No config files found in {training_config_dir}/")
        return None

    choices = [f.stem for f in config_files] + ["Cancel"]

    selected = prompts.select(
        ctx,
        "Select configuration:",
        choices=choices,
    )

    if selected is None or selected == "Cancel":
        return None

    config_path = training_config_dir / f"{selected}.yaml"
    config = load_config(config_path)

    edit = prompts.confirm(
        ctx,
        "Edit configuration before running?",
        default=False,
    )

    if edit:
        config = edit_config(ctx, config)

    return config


def edit_config(ctx: CliContext, config: Config) -> Config:
    """
    Interactive config editor with multiple categories.

    Args:
        ctx: CLI context
        config: Config to edit

    Returns:
        Modified config
    """
    ui.header("Edit Configuration")

    choices = [
        "Training Parameters (iterations, checkpoints, etc.)",
        "Game Settings (stack size, blinds)",
        "Solver Settings (CFR+, Linear CFR, sampling)",
        "Action Abstraction (bet sizes, raise cap)",
        "Card Abstraction",
        "System Settings (seed, logging)",
        "Done",
    ]

    while True:
        category = prompts.select(
            ctx,
            "What would you like to edit?",
            choices=choices,
        )

        if category == "Done" or category is None:
            break

        print()
        if "Training Parameters" in category:
            config = _edit_training_params(ctx, config)
        elif "Game Settings" in category:
            config = _edit_game_settings(ctx, config)
        elif "Solver Settings" in category:
            config = _edit_solver_settings(ctx, config)
        elif "Action Abstraction" in category:
            config = _edit_action_abstraction(ctx, config)
        elif "Card Abstraction" in category:
            config = _edit_card_abstraction(ctx, config)
        elif "System Settings" in category:
            config = _edit_system_settings(ctx, config)

        print()

    return config


def _edit_training_params(ctx: CliContext, config: Config) -> Config:
    """Edit training parameters."""
    print("Training Parameters")
    print("-" * 40)

    iterations = prompts.prompt_int(
        ctx,
        "Number of iterations:",
        default=config.training.num_iterations,
        min_value=1,
    )
    checkpoint_freq = prompts.prompt_int(
        ctx,
        "Checkpoint frequency (save every N iterations):",
        default=config.training.checkpoint_frequency,
        min_value=1,
    )
    log_freq = prompts.prompt_int(
        ctx,
        "Log frequency (print progress every N iterations):",
        default=config.training.log_frequency,
        min_value=1,
    )
    iterations_per_worker = prompts.prompt_int(
        ctx,
        "Iterations per worker (batch size multiplier for parallel training):",
        default=config.training.iterations_per_worker,
        min_value=1,
    )
    verbose = prompts.confirm(
        ctx,
        "Verbose output?",
        default=config.training.verbose,
    )

    if None in (iterations, checkpoint_freq, log_freq, iterations_per_worker, verbose):
        return config

    return config.merge(
        {
            "training": {
                "num_iterations": iterations,
                "checkpoint_frequency": checkpoint_freq,
                "log_frequency": log_freq,
                "iterations_per_worker": iterations_per_worker,
                "verbose": verbose,
            }
        }
    )


def _edit_game_settings(ctx: CliContext, config: Config) -> Config:
    """Edit game settings."""
    print("Game Settings")
    print("-" * 40)

    stack = prompts.prompt_int(
        ctx,
        "Starting stack (BB units):",
        default=config.game.starting_stack,
        min_value=1,
    )
    small_blind = prompts.prompt_int(
        ctx,
        "Small blind:",
        default=config.game.small_blind,
        min_value=1,
    )
    big_blind = prompts.prompt_int(
        ctx,
        "Big blind:",
        default=config.game.big_blind,
        min_value=1,
    )

    if None in (stack, small_blind, big_blind):
        return config

    return config.merge(
        {
            "game": {
                "starting_stack": stack,
                "small_blind": small_blind,
                "big_blind": big_blind,
            }
        }
    )


def _edit_solver_settings(ctx: CliContext, config: Config) -> Config:
    """Edit solver settings."""
    print("Solver Settings")
    print("-" * 40)

    sampling = prompts.select(
        ctx,
        "Sampling method:",
        choices=[
            "external (lower variance, slower)",
            "outcome (higher variance, faster)",
        ],
        default="outcome (higher variance, faster)"
        if config.solver.sampling_method == "outcome"
        else "external (lower variance, slower)",
    )
    cfr_plus = prompts.confirm(
        ctx,
        "Use CFR+? (100x faster convergence)",
        default=config.solver.cfr_plus,
    )
    linear_cfr = prompts.confirm(
        ctx,
        "Use Linear CFR? (2-3x additional speedup)",
        default=config.solver.linear_cfr,
    )

    if None in (sampling, cfr_plus, linear_cfr):
        return config

    sampling_method = "outcome" if "outcome" in sampling else "external"

    return config.merge(
        {
            "solver": {
                "sampling_method": sampling_method,
                "cfr_plus": cfr_plus,
                "linear_cfr": linear_cfr,
            }
        }
    )


def _edit_action_abstraction(ctx: CliContext, config: Config) -> Config:
    """Edit action abstraction settings."""
    print("Action Abstraction")
    print("-" * 40)

    max_raises = prompts.prompt_int(
        ctx,
        "Max raises per street (prevents infinite trees):",
        default=config.action_abstraction.max_raises_per_street,
        min_value=1,
    )
    spr_threshold = prompts.prompt_float(
        ctx,
        "All-in SPR threshold (allow all-in when SPR < threshold):",
        default=config.action_abstraction.all_in_spr_threshold,
        min_value=0.0,
    )

    if None in (max_raises, spr_threshold):
        return config

    return config.merge(
        {
            "action_abstraction": {
                "max_raises_per_street": max_raises,
                "all_in_spr_threshold": spr_threshold,
            }
        }
    )


def _edit_card_abstraction(ctx: CliContext, config: Config) -> Config:
    """Edit card abstraction settings."""
    print("Card Abstraction")
    print("-" * 40)

    default_config = config.card_abstraction.config or "default_plus"

    config_name = prompts.select(
        ctx,
        "Combo abstraction config:",
        choices=["fast_test", "default", "default_plus", "production"],
        default=default_config
        if default_config in ["fast_test", "default", "default_plus", "production"]
        else "default_plus",
    )

    if config_name is None:
        return config

    config = config.merge({"card_abstraction": {"config": config_name}})

    # Check if abstraction exists
    base_path = ctx.base_dir / "data" / "combo_abstraction"

    abstraction_found = False
    if base_path.exists():
        for path in base_path.iterdir():
            if path.is_dir() and (path / "combo_abstraction.pkl").exists():
                abstraction_found = True
                break

    if not abstraction_found:
        ui.warn("Warning: No combo abstraction found.")
        precompute = prompts.confirm(
            ctx,
            "Would you like to precompute combo abstraction now?",
            default=True,
        )

        if precompute:
            from src.cli.flows.combo import handle_combo_precompute

            handle_combo_precompute(ctx)

    return config


def _edit_system_settings(ctx: CliContext, config: Config) -> Config:
    """Edit system settings."""
    print("System Settings")
    print("-" * 40)

    def _validate_seed(value: str) -> bool | str:
        if not value.strip():
            return True
        if value.isdigit():
            return True
        return "Enter a whole number"

    seed_text = prompts.text(
        ctx,
        "Random seed (leave blank for random):",
        default=str(config.system.seed) if config.system.seed is not None else "",
        validate=_validate_seed,
    )
    log_level = prompts.select(
        ctx,
        "Log level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=config.system.log_level,
    )
    checkpoint_enabled = prompts.confirm(
        ctx,
        "Enable checkpointing?",
        default=config.storage.checkpoint_enabled,
    )

    if seed_text is None or log_level is None or checkpoint_enabled is None:
        return config

    seed = int(seed_text) if seed_text.strip() else None

    return config.merge(
        {
            "system": {
                "seed": seed,
                "log_level": log_level,
            },
            "storage": {
                "checkpoint_enabled": checkpoint_enabled,
            },
        }
    )
