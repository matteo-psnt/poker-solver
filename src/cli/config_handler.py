"""Configuration handling for CLI."""

from pathlib import Path
from typing import Any, Dict, Optional

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

    # Convert to dict for editing (since Config is frozen)
    overrides: Dict[str, Any] = {}

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
            _edit_training_params(config, overrides, custom_style)
        elif "Game Settings" in category:
            _edit_game_settings(config, overrides, custom_style)
        elif "Solver Settings" in category:
            _edit_solver_settings(config, overrides, custom_style)
        elif "Action Abstraction" in category:
            _edit_action_abstraction(config, overrides, custom_style)
        elif "Card Abstraction" in category:
            _edit_card_abstraction(config, overrides, custom_style)
        elif "System Settings" in category:
            _edit_system_settings(config, overrides, custom_style)

        print()

    # Create new config with overrides applied
    if overrides:
        from src.utils.config_loader import _merge_config

        return _merge_config(config, overrides)
    return config


def _edit_training_params(config: Config, overrides: Dict[str, Any], custom_style) -> None:
    """Edit training parameters (updates overrides dict in-place)."""
    print("Training Parameters")
    print("-" * 40)

    # Get current values (with any pending overrides)
    current_training = overrides.get("training", {})

    iterations = questionary.text(
        "Number of iterations:",
        default=str(current_training.get("num_iterations", config.training.num_iterations)),
        style=custom_style,
    ).ask()

    checkpoint_freq = questionary.text(
        "Checkpoint frequency (save every N iterations):",
        default=str(
            current_training.get("checkpoint_frequency", config.training.checkpoint_frequency)
        ),
        style=custom_style,
    ).ask()

    log_freq = questionary.text(
        "Log frequency (print progress every N iterations):",
        default=str(current_training.get("log_frequency", config.training.log_frequency)),
        style=custom_style,
    ).ask()

    iterations_per_worker = questionary.text(
        "Iterations per worker (batch size multiplier for parallel training):",
        default=str(
            current_training.get("iterations_per_worker", config.training.iterations_per_worker)
        ),
        style=custom_style,
    ).ask()

    verbose = questionary.confirm(
        "Verbose output?",
        default=current_training.get("verbose", config.training.verbose),
        style=custom_style,
    ).ask()

    # Update overrides
    if "training" not in overrides:
        overrides["training"] = {}
    overrides["training"]["num_iterations"] = int(iterations)
    overrides["training"]["checkpoint_frequency"] = int(checkpoint_freq)
    overrides["training"]["log_frequency"] = int(log_freq)
    overrides["training"]["iterations_per_worker"] = int(iterations_per_worker)
    overrides["training"]["verbose"] = verbose


def _edit_game_settings(config: Config, overrides: Dict[str, Any], custom_style) -> None:
    """Edit game settings (updates overrides dict in-place)."""
    print("Game Settings")
    print("-" * 40)

    current_game = overrides.get("game", {})

    stack = questionary.text(
        "Starting stack (BB units):",
        default=str(current_game.get("starting_stack", config.game.starting_stack)),
        style=custom_style,
    ).ask()

    small_blind = questionary.text(
        "Small blind:",
        default=str(current_game.get("small_blind", config.game.small_blind)),
        style=custom_style,
    ).ask()

    big_blind = questionary.text(
        "Big blind:",
        default=str(current_game.get("big_blind", config.game.big_blind)),
        style=custom_style,
    ).ask()

    if "game" not in overrides:
        overrides["game"] = {}
    overrides["game"]["starting_stack"] = int(stack)
    overrides["game"]["small_blind"] = int(small_blind)
    overrides["game"]["big_blind"] = int(big_blind)


def _edit_solver_settings(config: Config, overrides: Dict[str, Any], custom_style) -> None:
    """Edit solver settings (updates overrides dict in-place)."""
    print("Solver Settings")
    print("-" * 40)

    current_solver = overrides.get("solver", {})

    sampling = questionary.select(
        "Sampling method:",
        choices=[
            "external (lower variance, slower)",
            "outcome (higher variance, faster)",
        ],
        default="outcome (higher variance, faster)"
        if current_solver.get("sampling_method", config.solver.sampling_method) == "outcome"
        else "external (lower variance, slower)",
        style=custom_style,
    ).ask()

    cfr_plus = questionary.confirm(
        "Use CFR+? (100x faster convergence)",
        default=current_solver.get("cfr_plus", config.solver.cfr_plus),
        style=custom_style,
    ).ask()

    linear_cfr = questionary.confirm(
        "Use Linear CFR? (2-3x additional speedup)",
        default=current_solver.get("linear_cfr", config.solver.linear_cfr),
        style=custom_style,
    ).ask()

    sampling_method = "outcome" if "outcome" in sampling else "external"

    if "solver" not in overrides:
        overrides["solver"] = {}
    overrides["solver"]["sampling_method"] = sampling_method
    overrides["solver"]["cfr_plus"] = cfr_plus
    overrides["solver"]["linear_cfr"] = linear_cfr


def _edit_action_abstraction(config: Config, overrides: Dict[str, Any], custom_style) -> None:
    """Edit action abstraction settings (updates overrides dict in-place)."""
    print("Action Abstraction")
    print("-" * 40)

    current_action = overrides.get("action_abstraction", {})

    max_raises = questionary.text(
        "Max raises per street (prevents infinite trees):",
        default=str(
            current_action.get(
                "max_raises_per_street", config.action_abstraction.max_raises_per_street
            )
        ),
        style=custom_style,
    ).ask()

    spr_threshold = questionary.text(
        "All-in SPR threshold (allow all-in when SPR < threshold):",
        default=str(
            current_action.get(
                "all_in_spr_threshold", config.action_abstraction.all_in_spr_threshold
            )
        ),
        style=custom_style,
    ).ask()

    edit_bet_sizes = questionary.confirm(
        "Edit bet sizes? (Advanced - not recommended)",
        default=False,
        style=custom_style,
    ).ask()

    if "action_abstraction" not in overrides:
        overrides["action_abstraction"] = {}
    overrides["action_abstraction"]["max_raises_per_street"] = int(max_raises)
    overrides["action_abstraction"]["all_in_spr_threshold"] = float(spr_threshold)

    if edit_bet_sizes:
        print("\n[!] Warning: Editing bet sizes requires understanding poker theory.")
        print("    Default sizes are based on solver research.")
        print("    Press Enter to keep current values.\n")

        # Get current config as dict to access nested bet sizes
        config_dict = config.to_dict()
        action_dict = config_dict.get("action_abstraction", {})

        # Preflop
        preflop_raises = action_dict.get("preflop", {}).get("raises", [2.5, 3.5, 5.0])
        preflop_str = questionary.text(
            "Preflop raise sizes (BB units, comma-separated):",
            default=",".join(map(str, preflop_raises)),
            style=custom_style,
        ).ask()
        if "preflop" not in overrides["action_abstraction"]:
            overrides["action_abstraction"]["preflop"] = {}
        overrides["action_abstraction"]["preflop"]["raises"] = [
            float(x.strip()) for x in preflop_str.split(",")
        ]

        # Flop
        flop_bets = action_dict.get("postflop", {}).get("flop", {}).get("bets", [0.33, 0.66, 1.25])
        flop_str = questionary.text(
            "Flop bet sizes (pot fractions, comma-separated):",
            default=",".join(map(str, flop_bets)),
            style=custom_style,
        ).ask()
        if "postflop" not in overrides["action_abstraction"]:
            overrides["action_abstraction"]["postflop"] = {}
        if "flop" not in overrides["action_abstraction"]["postflop"]:
            overrides["action_abstraction"]["postflop"]["flop"] = {}
        overrides["action_abstraction"]["postflop"]["flop"]["bets"] = [
            float(x.strip()) for x in flop_str.split(",")
        ]

        # Turn
        turn_bets = action_dict.get("postflop", {}).get("turn", {}).get("bets", [0.50, 1.0, 1.5])
        turn_str = questionary.text(
            "Turn bet sizes (pot fractions, comma-separated):",
            default=",".join(map(str, turn_bets)),
            style=custom_style,
        ).ask()
        if "turn" not in overrides["action_abstraction"]["postflop"]:
            overrides["action_abstraction"]["postflop"]["turn"] = {}
        overrides["action_abstraction"]["postflop"]["turn"]["bets"] = [
            float(x.strip()) for x in turn_str.split(",")
        ]

        # River
        river_bets = action_dict.get("postflop", {}).get("river", {}).get("bets", [0.50, 1.0, 2.0])
        river_str = questionary.text(
            "River bet sizes (pot fractions, comma-separated):",
            default=",".join(map(str, river_bets)),
            style=custom_style,
        ).ask()
        if "river" not in overrides["action_abstraction"]["postflop"]:
            overrides["action_abstraction"]["postflop"]["river"] = {}
        overrides["action_abstraction"]["postflop"]["river"]["bets"] = [
            float(x.strip()) for x in river_str.split(",")
        ]


def _edit_card_abstraction(config: Config, overrides: Dict[str, Any], custom_style) -> None:
    """Edit card abstraction settings (updates overrides dict in-place)."""
    print("Card Abstraction")
    print("-" * 40)

    # Card abstraction is stored as dict in YAML (not in schema dataclass yet)
    config_dict = config.to_dict()
    card_config = config_dict.get("card_abstraction", {})
    current_card = overrides.get("card_abstraction", {})

    default_config = current_card.get("config", card_config.get("config", "default_plus"))

    config_name = questionary.select(
        "Combo abstraction config:",
        choices=["fast_test", "default", "default_plus", "production"],
        default=default_config
        if default_config in ["fast_test", "default", "default_plus", "production"]
        else "default_plus",
        style=custom_style,
    ).ask()

    if "card_abstraction" not in overrides:
        overrides["card_abstraction"] = {}
    overrides["card_abstraction"]["config"] = config_name

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


def _edit_system_settings(config: Config, overrides: Dict[str, Any], custom_style) -> None:
    """Edit system settings (updates overrides dict in-place)."""
    print("System Settings")
    print("-" * 40)

    current_system = overrides.get("system", {})
    current_storage = overrides.get("storage", {})

    current_seed = current_system.get("seed", config.system.seed)
    seed = questionary.text(
        "Random seed (leave blank for random):",
        default=str(current_seed) if current_seed is not None else "",
        style=custom_style,
    ).ask()

    log_level = questionary.select(
        "Log level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=current_system.get("log_level", config.system.log_level),
        style=custom_style,
    ).ask()

    checkpoint_enabled = questionary.confirm(
        "Enable checkpointing?",
        default=current_storage.get("checkpoint_enabled", config.storage.checkpoint_enabled),
        style=custom_style,
    ).ask()

    if "system" not in overrides:
        overrides["system"] = {}
    if seed.strip():
        overrides["system"]["seed"] = int(seed)
    else:
        overrides["system"]["seed"] = None

    overrides["system"]["log_level"] = log_level

    if "storage" not in overrides:
        overrides["storage"] = {}
    overrides["storage"]["checkpoint_enabled"] = checkpoint_enabled
