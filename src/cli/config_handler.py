"""Configuration handling for CLI."""

from pathlib import Path
from typing import Optional

import questionary

from src.utils.config import Config
from src.utils.config_loader import load_config


def select_config(config_dir: Path, custom_style) -> Optional[Config]:
    """
    Select and optionally edit a config file.

    Args:
        config_dir: Base config directory (e.g., "config/")
        custom_style: Questionary style

    Returns:
        Loaded Config object or None if cancelled
    """
    training_config_dir = config_dir / "training"
    config_files = sorted(training_config_dir.glob("*.yaml"))

    if not config_files:
        print(f"[ERROR] No config files found in {training_config_dir}/")
        return None

    choices = [f.stem for f in config_files] + ["Cancel"]

    selected = questionary.select(
        "Select configuration:",
        choices=choices,
        style=custom_style,
    ).ask()

    if selected == "Cancel" or selected is None:
        return None

    config_path = training_config_dir / f"{selected}.yaml"
    config = load_config(config_path)

    edit = questionary.confirm(
        "Edit configuration before running?",
        default=False,
        style=custom_style,
    ).ask()

    if edit:
        config = edit_config(config, custom_style)

    return config


def edit_config(config: Config, custom_style) -> Config:
    """
    Interactive config editor with multiple categories.

    Args:
        config: Config to edit
        custom_style: Questionary style

    Returns:
        Modified config
    """
    print("\nEdit Configuration")
    print("=" * 60)

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
        category = questionary.select(
            "What would you like to edit?",
            choices=choices,
            style=custom_style,
        ).ask()

        if category == "Done" or category is None:
            break

        print()
        if "Training Parameters" in category:
            config = _edit_training_params(config, custom_style)
        elif "Game Settings" in category:
            config = _edit_game_settings(config, custom_style)
        elif "Solver Settings" in category:
            config = _edit_solver_settings(config, custom_style)
        elif "Action Abstraction" in category:
            config = _edit_action_abstraction(config, custom_style)
        elif "Card Abstraction" in category:
            config = _edit_card_abstraction(config, custom_style)
        elif "System Settings" in category:
            config = _edit_system_settings(config, custom_style)

        print()

    return config


def _edit_training_params(config: Config, custom_style) -> Config:
    """Edit training parameters."""
    print("Training Parameters")
    print("-" * 40)

    iterations = questionary.text(
        "Number of iterations:",
        default=str(config.training.num_iterations),
        style=custom_style,
    ).ask()

    checkpoint_freq = questionary.text(
        "Checkpoint frequency (save every N iterations):",
        default=str(config.training.checkpoint_frequency),
        style=custom_style,
    ).ask()

    log_freq = questionary.text(
        "Log frequency (print progress every N iterations):",
        default=str(config.training.log_frequency),
        style=custom_style,
    ).ask()

    iterations_per_worker = questionary.text(
        "Iterations per worker (batch size multiplier for parallel training):",
        default=str(config.training.iterations_per_worker),
        style=custom_style,
    ).ask()

    verbose = questionary.confirm(
        "Verbose output?",
        default=config.training.verbose,
        style=custom_style,
    ).ask()

    return config.merge(
        {
            "training": {
                "num_iterations": int(iterations),
                "checkpoint_frequency": int(checkpoint_freq),
                "log_frequency": int(log_freq),
                "iterations_per_worker": int(iterations_per_worker),
                "verbose": verbose,
            }
        }
    )


def _edit_game_settings(config: Config, custom_style) -> Config:
    """Edit game settings."""
    print("Game Settings")
    print("-" * 40)

    stack = questionary.text(
        "Starting stack (BB units):",
        default=str(config.game.starting_stack),
        style=custom_style,
    ).ask()

    small_blind = questionary.text(
        "Small blind:",
        default=str(config.game.small_blind),
        style=custom_style,
    ).ask()

    big_blind = questionary.text(
        "Big blind:",
        default=str(config.game.big_blind),
        style=custom_style,
    ).ask()

    return config.merge(
        {
            "game": {
                "starting_stack": int(stack),
                "small_blind": int(small_blind),
                "big_blind": int(big_blind),
            }
        }
    )


def _edit_solver_settings(config: Config, custom_style) -> Config:
    """Edit solver settings."""
    print("Solver Settings")
    print("-" * 40)

    sampling = questionary.select(
        "Sampling method:",
        choices=[
            "external (lower variance, slower)",
            "outcome (higher variance, faster)",
        ],
        default="outcome (higher variance, faster)"
        if config.solver.sampling_method == "outcome"
        else "external (lower variance, slower)",
        style=custom_style,
    ).ask()

    cfr_plus = questionary.confirm(
        "Use CFR+? (100x faster convergence)",
        default=config.solver.cfr_plus,
        style=custom_style,
    ).ask()

    linear_cfr = questionary.confirm(
        "Use Linear CFR? (2-3x additional speedup)",
        default=config.solver.linear_cfr,
        style=custom_style,
    ).ask()

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


def _edit_action_abstraction(config: Config, custom_style) -> Config:
    """Edit action abstraction settings."""
    print("Action Abstraction")
    print("-" * 40)

    max_raises = questionary.text(
        "Max raises per street (prevents infinite trees):",
        default=str(config.action_abstraction.max_raises_per_street),
        style=custom_style,
    ).ask()

    spr_threshold = questionary.text(
        "All-in SPR threshold (allow all-in when SPR < threshold):",
        default=str(config.action_abstraction.all_in_spr_threshold),
        style=custom_style,
    ).ask()

    return config.merge(
        {
            "action_abstraction": {
                "max_raises_per_street": int(max_raises),
                "all_in_spr_threshold": float(spr_threshold),
            }
        }
    )


def _edit_card_abstraction(config: Config, custom_style) -> Config:
    """Edit card abstraction settings."""
    print("Card Abstraction")
    print("-" * 40)

    default_config = config.card_abstraction.config or "default_plus"

    config_name = questionary.select(
        "Combo abstraction config:",
        choices=["fast_test", "default", "default_plus", "production"],
        default=default_config
        if default_config in ["fast_test", "default", "default_plus", "production"]
        else "default_plus",
        style=custom_style,
    ).ask()

    config = config.merge({"card_abstraction": {"config": config_name}})

    # Check if abstraction exists
    base_path = Path("data/combo_abstraction")

    abstraction_found = False
    if base_path.exists():
        for path in base_path.iterdir():
            if path.is_dir() and (path / "combo_abstraction.pkl").exists():
                abstraction_found = True
                break

    if not abstraction_found:
        print("\n[!] Warning: No combo abstraction found.")
        precompute = questionary.confirm(
            "Would you like to precompute combo abstraction now?",
            default=True,
            style=custom_style,
        ).ask()

        if precompute:
            from src.cli.combo_handler import handle_combo_precompute

            handle_combo_precompute()

    return config


def _edit_system_settings(config: Config, custom_style) -> Config:
    """Edit system settings."""
    print("System Settings")
    print("-" * 40)

    seed = questionary.text(
        "Random seed (leave blank for random):",
        default=str(config.system.seed) if config.system.seed is not None else "",
        style=custom_style,
    ).ask()

    log_level = questionary.select(
        "Log level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=config.system.log_level,
        style=custom_style,
    ).ask()

    checkpoint_enabled = questionary.confirm(
        "Enable checkpointing?",
        default=config.storage.checkpoint_enabled,
        style=custom_style,
    ).ask()

    return config.merge(
        {
            "system": {
                "seed": int(seed) if seed.strip() else None,
                "log_level": log_level,
            },
            "storage": {
                "checkpoint_enabled": checkpoint_enabled,
            },
        }
    )
