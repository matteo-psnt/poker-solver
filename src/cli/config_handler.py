"""Configuration handling for CLI."""

from pathlib import Path
from typing import Optional

import questionary

from src.utils.config import Config


def select_config(config_dir: Path, custom_style) -> Optional[Config]:
    """
    Select and optionally edit a config file.

    Args:
        config_dir: Base config directory (e.g., "config/")
        custom_style: Questionary style

    Returns:
        Loaded Config object or None if cancelled
    """
    # Look for training configs in config/training/ subdirectory
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
    config = Config.from_file(config_path)

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

    # Select what to edit
    choices = [
        "Training Parameters (iterations, checkpoints, etc.)",
        "Game Settings (stack size, blinds)",
        "Solver Settings (CFR+, Linear CFR, sampling)",
        "Action Abstraction (bet sizes, raise cap)",
        "Card Abstraction",
        "System Settings (seed, logging)",
        "Done - Save Changes",
    ]

    while True:
        category = questionary.select(
            "What would you like to edit?",
            choices=choices,
            style=custom_style,
        ).ask()

        if category == "Done - Save Changes" or category is None:
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
        default=str(config.get("training.num_iterations", 100000)),
        style=custom_style,
    ).ask()

    checkpoint_freq = questionary.text(
        "Checkpoint frequency (save every N iterations):",
        default=str(config.get("training.checkpoint_frequency", 2000)),
        style=custom_style,
    ).ask()

    log_freq = questionary.text(
        "Log frequency (print progress every N iterations):",
        default=str(config.get("training.log_frequency", 1000)),
        style=custom_style,
    ).ask()

    iterations_per_worker = questionary.text(
        "Iterations per worker (batch size multiplier for parallel training):",
        default=str(config.get("training.iterations_per_worker", 100)),
        style=custom_style,
    ).ask()

    verbose = questionary.confirm(
        "Verbose output?",
        default=config.get("training.verbose", True),
        style=custom_style,
    ).ask()

    config.set("training.num_iterations", int(iterations))
    config.set("training.checkpoint_frequency", int(checkpoint_freq))
    config.set("training.log_frequency", int(log_freq))
    config.set("training.iterations_per_worker", int(iterations_per_worker))
    config.set("training.verbose", verbose)

    return config


def _edit_game_settings(config: Config, custom_style) -> Config:
    """Edit game settings."""
    print("Game Settings")
    print("-" * 40)

    stack = questionary.text(
        "Starting stack (BB units):",
        default=str(config.get("game.starting_stack", 200)),
        style=custom_style,
    ).ask()

    small_blind = questionary.text(
        "Small blind:",
        default=str(config.get("game.small_blind", 1)),
        style=custom_style,
    ).ask()

    big_blind = questionary.text(
        "Big blind:",
        default=str(config.get("game.big_blind", 2)),
        style=custom_style,
    ).ask()

    config.set("game.starting_stack", int(stack))
    config.set("game.small_blind", int(small_blind))
    config.set("game.big_blind", int(big_blind))

    return config


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
        if config.get("solver.sampling_method", "outcome") == "outcome"
        else "external (lower variance, slower)",
        style=custom_style,
    ).ask()

    cfr_plus = questionary.confirm(
        "Use CFR+? (100x faster convergence)",
        default=config.get("solver.cfr_plus", True),
        style=custom_style,
    ).ask()

    linear_cfr = questionary.confirm(
        "Use Linear CFR? (2-3x additional speedup)",
        default=config.get("solver.linear_cfr", True),
        style=custom_style,
    ).ask()

    sampling_method = "outcome" if "outcome" in sampling else "external"
    config.set("solver.sampling_method", sampling_method)
    config.set("solver.cfr_plus", cfr_plus)
    config.set("solver.linear_cfr", linear_cfr)

    return config


def _edit_action_abstraction(config: Config, custom_style) -> Config:
    """Edit action abstraction settings."""
    print("Action Abstraction")
    print("-" * 40)

    max_raises = questionary.text(
        "Max raises per street (prevents infinite trees):",
        default=str(config.get("action_abstraction.max_raises_per_street", 4)),
        style=custom_style,
    ).ask()

    spr_threshold = questionary.text(
        "All-in SPR threshold (allow all-in when SPR < threshold):",
        default=str(config.get("action_abstraction.all_in_spr_threshold", 2.0)),
        style=custom_style,
    ).ask()

    edit_bet_sizes = questionary.confirm(
        "Edit bet sizes? (Advanced - not recommended)",
        default=False,
        style=custom_style,
    ).ask()

    config.set("action_abstraction.max_raises_per_street", int(max_raises))
    config.set("action_abstraction.all_in_spr_threshold", float(spr_threshold))

    if edit_bet_sizes:
        print("\n[!] Warning: Editing bet sizes requires understanding poker theory.")
        print("    Default sizes are based on solver research.")
        print("    Press Enter to keep current values.\n")

        # Preflop
        preflop_raises = config.get("action_abstraction.preflop.raises", [2.5, 3.5, 5.0])
        preflop_str = questionary.text(
            "Preflop raise sizes (BB units, comma-separated):",
            default=",".join(map(str, preflop_raises)),
            style=custom_style,
        ).ask()
        config.set(
            "action_abstraction.preflop.raises",
            [float(x.strip()) for x in preflop_str.split(",")],
        )

        # Flop
        flop_bets = config.get("action_abstraction.postflop.flop.bets", [0.33, 0.66, 1.25])
        flop_str = questionary.text(
            "Flop bet sizes (pot fractions, comma-separated):",
            default=",".join(map(str, flop_bets)),
            style=custom_style,
        ).ask()
        config.set(
            "action_abstraction.postflop.flop.bets",
            [float(x.strip()) for x in flop_str.split(",")],
        )

        # Turn
        turn_bets = config.get("action_abstraction.postflop.turn.bets", [0.50, 1.0, 1.5])
        turn_str = questionary.text(
            "Turn bet sizes (pot fractions, comma-separated):",
            default=",".join(map(str, turn_bets)),
            style=custom_style,
        ).ask()
        config.set(
            "action_abstraction.postflop.turn.bets",
            [float(x.strip()) for x in turn_str.split(",")],
        )

        # River
        river_bets = config.get("action_abstraction.postflop.river.bets", [0.50, 1.0, 2.0])
        river_str = questionary.text(
            "River bet sizes (pot fractions, comma-separated):",
            default=",".join(map(str, river_bets)),
            style=custom_style,
        ).ask()
        config.set(
            "action_abstraction.postflop.river.bets",
            [float(x.strip()) for x in river_str.split(",")],
        )

    return config


def _edit_card_abstraction(config: Config, custom_style) -> Config:
    """Edit card abstraction settings."""
    print("Card Abstraction")
    print("-" * 40)

    default_config = config.get("card_abstraction.config", "default_plus")

    config_name = questionary.select(
        "Combo abstraction config:",
        choices=["fast_test", "default", "default_plus", "production"],
        default=default_config
        if default_config in ["fast_test", "default", "default_plus", "production"]
        else "default_plus",
        style=custom_style,
    ).ask()

    config.set("card_abstraction.config", config_name)

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
        default=str(config.get("system.seed", "")) if config.get("system.seed") is not None else "",
        style=custom_style,
    ).ask()

    log_level = questionary.select(
        "Log level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=config.get("system.log_level", "INFO"),
        style=custom_style,
    ).ask()

    checkpoint_enabled = questionary.confirm(
        "Enable checkpointing?",
        default=config.get("storage.checkpoint_enabled", True),
        style=custom_style,
    ).ask()

    if seed.strip():
        config.set("system.seed", int(seed))
    else:
        config.set("system.seed", None)

    config.set("system.log_level", log_level)
    config.set("storage.checkpoint_enabled", checkpoint_enabled)

    return config
