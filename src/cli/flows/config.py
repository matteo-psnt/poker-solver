"""Configuration handling for CLI."""

from src.cli.flows.combo_precompute import handle_combo_precompute
from src.cli.ui import prompts, ui
from src.cli.ui.context import CliContext
from src.utils.config import Config
from src.utils.config_loader import load_config


def select_config(ctx: CliContext) -> Config | None:
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
        "Action Model & Resolver (templates, raise cap)",
        "Card Abstraction",
        "Storage Settings (capacity, compression)",
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
        elif "Action Model" in category:
            config = _edit_action_abstraction(ctx, config)
        elif "Card Abstraction" in category:
            config = _edit_card_abstraction(ctx, config)
        elif "Storage Settings" in category:
            config = _edit_storage_settings(ctx, config)
        elif "System Settings" in category:
            config = _edit_system_settings(ctx, config)

        print()

    return config


def _list_abstraction_configs(ctx: CliContext) -> list[str]:
    """List available card abstraction config names from config/abstraction."""
    abstraction_config_dir = ctx.config_dir / "abstraction"
    if not abstraction_config_dir.exists():
        return []

    return sorted(
        config_file.stem
        for config_file in abstraction_config_dir.glob("*.yaml")
        if config_file.is_file()
    )


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

    runs_dir = prompts.text(
        ctx,
        "Output directory for training runs:",
        default=config.training.runs_dir,
    )

    if None in (iterations, checkpoint_freq, iterations_per_worker, verbose, runs_dir):
        return config

    return config.merge(
        {
            "training": {
                "num_iterations": iterations,
                "checkpoint_frequency": checkpoint_freq,
                "iterations_per_worker": iterations_per_worker,
                "verbose": verbose,
                "runs_dir": runs_dir,
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
    """Edit action model and resolver settings."""
    print("Action Model & Resolver")
    print("-" * 40)

    max_raises = prompts.prompt_int(
        ctx,
        "Max raises per street (prevents infinite trees):",
        default=config.resolver.max_raises_per_street,
        min_value=1,
    )
    jam_spr_cutoff = prompts.prompt_float(
        ctx,
        "Jam SPR cutoff for 'jam_low_spr' templates:",
        default=config.action_model.all_in_spr_threshold,
        min_value=0.0,
    )

    # Ask if user wants to customize bet sizes
    customize_bets = prompts.confirm(
        ctx,
        "Customize bet sizes? (Advanced)",
        default=False,
    )

    if None in (max_raises, jam_spr_cutoff):
        return config

    existing_buckets = list(config.action_model.spr_buckets)
    if len(existing_buckets) < 2:
        existing_buckets = [jam_spr_cutoff, max(jam_spr_cutoff, 6.0)]
    else:
        existing_buckets[0] = jam_spr_cutoff

    merge_dict = {
        "resolver": {
            "max_raises_per_street": max_raises,
        },
        "action_model": {
            "spr_buckets": existing_buckets,
        },
    }

    if customize_bets:
        ui.info("\nPreflop Raise Sizes (BB units):")
        default_preflop = [
            float(v)
            for v in config.action_model.preflop_templates.get("sb_first_in", [])
            if isinstance(v, (int, float))
        ]
        preflop_raises = _edit_list_of_floats(
            ctx,
            "Enter raise sizes separated by commas",
            default=default_preflop or [2.0, 2.5],
        )

        ui.info("\nPostflop Bet Sizes (pot fractions):")
        first_aggressive = [
            float(v)
            for v in config.action_model.postflop_templates.get("first_aggressive", [])
            if isinstance(v, (int, float))
        ]
        flop_bets = _edit_list_of_floats(
            ctx,
            "Flop bet sizes (e.g. 0.33, 0.66, 1.25):",
            default=first_aggressive or [0.33, 0.75, 1.25],
        )
        turn_bets = _edit_list_of_floats(
            ctx,
            "Turn bet sizes (e.g. 0.50, 1.0, 1.5):",
            default=first_aggressive or [0.5, 1.0, 1.5],
        )
        river_bets = _edit_list_of_floats(
            ctx,
            "River bet sizes (e.g. 0.50, 1.0, 2.0):",
            default=first_aggressive or [0.5, 1.0, 2.0],
        )

        preflop_templates = dict(config.action_model.preflop_templates)
        postflop_templates = dict(config.action_model.postflop_templates)

        if preflop_raises:
            passive = [
                token
                for token in preflop_templates.get("sb_first_in", [])
                if isinstance(token, str) and token in {"fold", "call", "limp"}
            ]
            preflop_templates["sb_first_in"] = passive + preflop_raises
        if flop_bets and turn_bets and river_bets:
            # Keep one shared aggressive template for simplicity in the editor.
            postflop_templates["first_aggressive"] = flop_bets

        merge_dict["action_model"]["preflop_templates"] = preflop_templates
        merge_dict["action_model"]["postflop_templates"] = postflop_templates

    return config.merge(merge_dict)


def _edit_list_of_floats(ctx: CliContext, prompt: str, default: list[float]) -> list[float] | None:
    """Helper to edit a list of floats."""
    default_str = ", ".join(str(x) for x in default)

    def validate(value: str) -> bool | str:
        try:
            parts = [float(x.strip()) for x in value.split(",")]
            if len(parts) == 0:
                return "Need at least one value"
            if any(x <= 0 for x in parts):
                return "All values must be positive"
            return True
        except ValueError:
            return "Invalid number format"

    result = prompts.text(ctx, prompt, default=default_str, validate=validate)
    if result is None:
        return None

    try:
        return [float(x.strip()) for x in result.split(",")]
    except ValueError:
        return None


def _edit_storage_settings(ctx: CliContext, config: Config) -> Config:
    """Edit storage settings."""
    print("Storage Settings")
    print("-" * 40)

    ui.info("These settings affect memory usage and checkpoint performance.")

    initial_capacity = prompts.prompt_int(
        ctx,
        "Initial infoset capacity (grows automatically):",
        default=config.storage.initial_capacity,
        min_value=10_000,
    )

    max_actions = prompts.prompt_int(
        ctx,
        "Max actions per infoset:",
        default=config.storage.max_actions,
        min_value=2,
    )

    zarr_compression = prompts.prompt_int(
        ctx,
        "Zarr compression level (1=fastest, 3=balanced, 9=smallest, 1-9):",
        default=config.storage.zarr_compression_level,
        min_value=1,
    )
    if zarr_compression is not None and zarr_compression > 9:
        print("⚠️  Warning: Compression level > 9 is unusual. Using 9.")
        zarr_compression = 9

    zarr_chunk = prompts.prompt_int(
        ctx,
        "Zarr chunk size (infosets per chunk, 10K-100K typical):",
        default=config.storage.zarr_chunk_size,
        min_value=1_000,
    )

    if None in (initial_capacity, max_actions, zarr_compression, zarr_chunk):
        return config

    return config.merge(
        {
            "storage": {
                "initial_capacity": initial_capacity,
                "max_actions": max_actions,
                "zarr_compression_level": zarr_compression,
                "zarr_chunk_size": zarr_chunk,
            }
        }
    )


def _edit_card_abstraction(ctx: CliContext, config: Config) -> Config:
    """Edit card abstraction settings."""
    print("Card Abstraction")
    print("-" * 40)

    available_configs = _list_abstraction_configs(ctx)
    if not available_configs:
        ui.error(f"No abstraction config files found in {ctx.config_dir / 'abstraction'}/")
        return config

    default_config = config.card_abstraction.config or "default"
    if default_config in available_configs:
        prompt_default = default_config
    elif "default" in available_configs:
        prompt_default = "default"
    else:
        prompt_default = available_configs[0]

    config_name = prompts.select(
        ctx,
        "Combo abstraction config:",
        choices=available_configs,
        default=prompt_default,
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
