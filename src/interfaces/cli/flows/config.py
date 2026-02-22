"""Configuration handling for CLI."""

from pydantic import ValidationError

from src.interfaces.cli.flows.combo_precompute import handle_combo_precompute
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.training.components import build_card_abstraction
from src.shared.config import Config
from src.shared.config_loader import load_config


def select_config(ctx: CliContext) -> Config | None:
    """
    Select and optionally edit a config file.

    Returns:
        Loaded Config object or None if cancelled.
    """
    training_config_dir = ctx.config_dir / "training"
    config_files = sorted(training_config_dir.glob("*.yaml"))

    if not config_files:
        ui.error(f"No config files found in {training_config_dir}/")
        return None

    choices = [f.stem for f in config_files] + ["Cancel"]

    selected = prompts.select(ctx, "Select configuration:", choices=choices)

    if selected is None or selected == "Cancel":
        return None

    config_path = training_config_dir / f"{selected}.yaml"
    config = load_config(config_path)

    edit = prompts.confirm(ctx, "Edit configuration before running?", default=False)

    if edit:
        config = edit_config(ctx, config)

    return config


def edit_config(ctx: CliContext, config: Config) -> Config:
    """
    Interactive config editor with multiple categories.

    Returns:
        Modified config (original is returned unchanged on cancel).
    """
    ui.header("Edit Configuration")

    choices = [
        "Training Parameters",
        "Game Settings",
        "Solver Settings",
        "Pruning",
        "Action Model & Resolver",
        "Card Abstraction",
        "Storage Settings",
        "System Settings",
        "Done",
    ]

    while True:
        category = prompts.select(ctx, "What would you like to edit?", choices=choices)

        if category == "Done" or category is None:
            break

        print()
        handler = {
            "Training Parameters": _edit_training_params,
            "Game Settings": _edit_game_settings,
            "Solver Settings": _edit_solver_settings,
            "Pruning": _edit_dcfr_pruning,
            "Action Model & Resolver": _edit_action_model,
            "Card Abstraction": _edit_card_abstraction,
            "Storage Settings": _edit_storage_settings,
            "System Settings": _edit_system_settings,
        }.get(category)

        if handler:
            config = handler(ctx, config)

        print()

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _list_abstraction_configs(ctx: CliContext) -> list[str]:
    """List available card abstraction config names from config/abstraction."""
    abstraction_config_dir = ctx.config_dir / "abstraction"
    if not abstraction_config_dir.exists():
        return []
    return sorted(f.stem for f in abstraction_config_dir.glob("*.yaml") if f.is_file())


def _try_merge(config: Config, overrides: dict) -> Config:
    """
    Attempt config.merge(overrides), printing a clear error and returning
    the original config unchanged if Pydantic validation fails.
    """
    try:
        return config.merge(overrides)
    except ValidationError as exc:
        ui.error("Invalid configuration — changes not applied:")
        for error in exc.errors():
            field = " → ".join(str(p) for p in error["loc"])
            print(f"  {field}: {error['msg']}")
        return config


# ---------------------------------------------------------------------------
# Section editors
# ---------------------------------------------------------------------------


def _edit_training_params(ctx: CliContext, config: Config) -> Config:
    print("Training Parameters")
    print("-" * 40)
    ui.info("Total iterations run × iterations_per_worker = total work per training session.")

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

    return _try_merge(
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


def _edit_game_settings(ctx: CliContext, config: Config) -> Config:
    print("Game Settings")
    print("-" * 40)

    stack = prompts.prompt_int(
        ctx,
        "Starting stack (BB units):",
        default=config.game.starting_stack,
        min_value=1,
    )
    if stack is None:
        return config

    small_blind = prompts.prompt_int(
        ctx,
        "Small blind (chips):",
        default=config.game.small_blind,
        min_value=1,
    )
    if small_blind is None:
        return config

    big_blind = prompts.prompt_int(
        ctx,
        "Big blind (chips):",
        default=config.game.big_blind,
        min_value=1,
    )
    if big_blind is None:
        return config

    return _try_merge(
        config,
        {
            "game": {
                "starting_stack": stack,
                "small_blind": small_blind,
                "big_blind": big_blind,
            }
        },
    )


def _edit_solver_settings(ctx: CliContext, config: Config) -> Config:
    print("Solver Settings")
    print("-" * 40)

    sampling = prompts.select(
        ctx,
        "Sampling method:",
        choices=[
            "external (lower variance, recommended for production)",
            "outcome (higher variance, faster per iteration)",
        ],
        default=(
            "outcome (higher variance, faster per iteration)"
            if config.solver.sampling_method == "outcome"
            else "external (lower variance, recommended for production)"
        ),
    )
    if sampling is None:
        return config

    cfr_plus = prompts.confirm(
        ctx,
        "Use CFR+? (floors regrets at 0 — ~100x faster convergence)",
        default=config.solver.cfr_plus,
    )
    if cfr_plus is None:
        return config

    weighting = prompts.select(
        ctx,
        "Iteration weighting:",
        choices=[
            "linear (weights later iterations more — 2–3x speedup, recommended)",
            "dcfr (Discounted CFR — Brown & Sandholm 2019)",
            "none (uniform weighting — vanilla CFR)",
        ],
        default={
            "linear": "linear (weights later iterations more — 2–3x speedup, recommended)",
            "dcfr": "dcfr (Discounted CFR — Brown & Sandholm 2019)",
            "none": "none (uniform weighting — vanilla CFR)",
        }.get(config.solver.iteration_weighting),
    )
    if weighting is None:
        return config

    iteration_weighting = weighting.split(" ")[0]  # extract "linear" / "dcfr" / "none"

    overrides: dict = {
        "solver": {
            "sampling_method": "outcome" if "outcome" in sampling else "external",
            "cfr_plus": cfr_plus,
            "iteration_weighting": iteration_weighting,
        }
    }

    if iteration_weighting == "dcfr":
        dcfr_alpha = prompts.prompt_float(
            ctx,
            "DCFR alpha — positive-regret discount exponent (recommended: 1.5):",
            default=config.solver.dcfr_alpha,
            min_value=0.001,
        )
        if dcfr_alpha is None:
            return config

        dcfr_beta = prompts.prompt_float(
            ctx,
            "DCFR beta — negative-regret discount exponent (recommended: 0.0):",
            default=config.solver.dcfr_beta,
            min_value=0.0,
        )
        if dcfr_beta is None:
            return config

        dcfr_gamma = prompts.prompt_float(
            ctx,
            "DCFR gamma — strategy discount exponent (recommended: 2.0):",
            default=config.solver.dcfr_gamma,
            min_value=0.001,
        )
        if dcfr_gamma is None:
            return config

        overrides["solver"].update(
            {
                "dcfr_alpha": dcfr_alpha,
                "dcfr_beta": dcfr_beta,
                "dcfr_gamma": dcfr_gamma,
            }
        )

    return _try_merge(config, overrides)


def _edit_dcfr_pruning(ctx: CliContext, config: Config) -> Config:
    print("Pruning")
    print("-" * 40)

    enable_pruning = prompts.confirm(
        ctx,
        "Enable regret-based pruning? (skips low-regret actions during training)",
        default=config.solver.enable_pruning,
    )
    if enable_pruning is None:
        return config

    overrides: dict = {"solver": {"enable_pruning": enable_pruning}}

    if enable_pruning:
        threshold = prompts.prompt_float(
            ctx,
            "Pruning threshold (actions with regret below −threshold are pruned):",
            default=config.solver.pruning_threshold,
            min_value=0.0,
        )
        if threshold is None:
            return config

        prune_start = prompts.prompt_int(
            ctx,
            "Start pruning after iteration:",
            default=config.solver.prune_start_iteration,
            min_value=1,
        )
        if prune_start is None:
            return config

        reactivate_freq = prompts.prompt_int(
            ctx,
            "Re-enable all pruned actions every N iterations:",
            default=config.solver.prune_reactivate_frequency,
            min_value=1,
        )
        if reactivate_freq is None:
            return config

        overrides["solver"].update(
            {
                "pruning_threshold": threshold,
                "prune_start_iteration": prune_start,
                "prune_reactivate_frequency": reactivate_freq,
            }
        )

    return _try_merge(config, overrides)


def _edit_action_model(ctx: CliContext, config: Config) -> Config:
    print("Action Model & Resolver")
    print("-" * 40)

    max_raises = prompts.prompt_int(
        ctx,
        "Max raises per street in the resolver subgame:",
        default=config.resolver.max_raises_per_street,
        min_value=1,
    )
    if max_raises is None:
        return config

    jam_spr_cutoff = prompts.prompt_float(
        ctx,
        "Jam SPR threshold (jam_low_spr fires when pot-to-stack ratio is below this):",
        default=config.action_model.jam_spr_threshold,
        min_value=0.0,
    )
    if jam_spr_cutoff is None:
        return config

    customize_bets = prompts.confirm(ctx, "Customise bet sizes? (Advanced)", default=False)
    if customize_bets is None:
        return config

    merge_dict: dict = {
        "resolver": {"max_raises_per_street": max_raises},
        "action_model": {"jam_spr_threshold": jam_spr_cutoff},
    }

    if customize_bets:
        ui.info("\nPreflop Raise Sizes (BB units, comma-separated):")
        default_preflop = [
            float(v)
            for v in config.action_model.preflop_templates.get("sb_first_in", [])
            if isinstance(v, (int, float))
        ]
        preflop_raises = _edit_list_of_floats(
            ctx,
            "Enter raise sizes (e.g. 2.5, 3.5, 5.0):",
            default=default_preflop or [2.5, 3.5, 5.0],
        )
        if preflop_raises is None:
            return config

        ui.info("\nPostflop Bet Sizes (pot fractions, comma-separated):")
        first_aggressive = [
            float(v)
            for v in config.action_model.postflop_templates.get("first_aggressive", [])
            if isinstance(v, (int, float))
        ]
        flop_bets = _edit_list_of_floats(
            ctx,
            "First-to-act bet sizes (e.g. 0.33, 0.66, 1.25):",
            default=first_aggressive or [0.33, 0.66, 1.25],
        )
        if flop_bets is None:
            return config

        preflop_templates = dict(config.action_model.preflop_templates)
        postflop_templates = dict(config.action_model.postflop_templates)

        passive = [
            t
            for t in preflop_templates.get("sb_first_in", [])
            if isinstance(t, str) and t in {"fold", "call", "limp"}
        ]
        preflop_templates["sb_first_in"] = passive + preflop_raises
        postflop_templates["first_aggressive"] = flop_bets

        merge_dict["action_model"]["preflop_templates"] = preflop_templates
        merge_dict["action_model"]["postflop_templates"] = postflop_templates

    return _try_merge(config, merge_dict)


def _edit_list_of_floats(ctx: CliContext, prompt: str, default: list[float]) -> list[float] | None:
    """Prompt for a comma-separated list of positive floats."""
    default_str = ", ".join(str(x) for x in default)

    def validate(value: str) -> bool | str:
        try:
            parts = [float(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            return "Invalid number — use comma-separated values like 0.33, 0.66, 1.25"
        if not parts:
            return "Enter at least one value"
        if any(x <= 0 for x in parts):
            return "All values must be positive"
        return True

    result = prompts.text(ctx, prompt, default=default_str, validate=validate)
    if result is None:
        return None
    try:
        return [float(x.strip()) for x in result.split(",") if x.strip()]
    except ValueError:
        return None


def _edit_storage_settings(ctx: CliContext, config: Config) -> Config:
    print("Storage Settings")
    print("-" * 40)
    ui.info("These settings affect memory usage and checkpoint I/O performance.")

    initial_capacity = prompts.prompt_int(
        ctx,
        "Initial infoset capacity (grows automatically if exceeded):",
        default=config.storage.initial_capacity,
        min_value=10_000,
    )
    if initial_capacity is None:
        return config

    max_actions = prompts.prompt_int(
        ctx,
        "Max actions stored per infoset:",
        default=config.storage.max_actions,
        min_value=2,
    )
    if max_actions is None:
        return config

    zarr_compression = prompts.prompt_int(
        ctx,
        "Zarr compression level (1=fastest I/O, 9=smallest files):",
        default=config.storage.zarr_compression_level,
        min_value=1,
        max_value=9,
    )
    if zarr_compression is None:
        return config

    zarr_chunk = prompts.prompt_int(
        ctx,
        "Zarr chunk size in infosets (10K–100K typical; larger = faster sequential reads):",
        default=config.storage.zarr_chunk_size,
        min_value=1_000,
    )
    if zarr_chunk is None:
        return config

    return _try_merge(
        config,
        {
            "storage": {
                "initial_capacity": initial_capacity,
                "max_actions": max_actions,
                "zarr_compression_level": zarr_compression,
                "zarr_chunk_size": zarr_chunk,
            }
        },
    )


def _edit_card_abstraction(ctx: CliContext, config: Config) -> Config:
    print("Card Abstraction")
    print("-" * 40)

    available_configs = _list_abstraction_configs(ctx)
    if not available_configs:
        ui.error(f"No abstraction config files found in {ctx.config_dir / 'abstraction'}/")
        return config

    default_config = config.card_abstraction.config or "default"
    prompt_default = (
        default_config
        if default_config in available_configs
        else ("default" if "default" in available_configs else available_configs[0])
    )

    config_name = prompts.select(
        ctx,
        "Combo abstraction config:",
        choices=available_configs,
        default=prompt_default,
    )
    if config_name is None:
        return config

    config = _try_merge(config, {"card_abstraction": {"config": config_name}})

    # Verify the selected abstraction actually exists and the hash matches.
    # build_card_abstraction uses the same resolver logic as training, so this
    # catches both "never precomputed" and "config changed since last precompute".
    try:
        build_card_abstraction(config)
    except FileNotFoundError:
        ui.warn(f"No precomputed abstraction found for '{config_name}'.")
        if prompts.confirm(ctx, "Run precomputation now?", default=True):
            handle_combo_precompute(ctx)
    except ValueError as exc:
        if "hash mismatch" in str(exc).lower():
            ui.warn(f"Abstraction for '{config_name}' was built with different parameters.")
            if prompts.confirm(ctx, "Recompute now?", default=True):
                handle_combo_precompute(ctx)
        else:
            ui.error(str(exc))

    return config


def _edit_system_settings(ctx: CliContext, config: Config) -> Config:
    print("System Settings")
    print("-" * 40)

    def _validate_seed(value: str) -> bool | str:
        if not value.strip():
            return True
        try:
            int(value)
            return True
        except ValueError:
            return "Enter a whole number, or leave blank for a random seed"

    seed_text = prompts.text(
        ctx,
        "Random seed (leave blank for a different seed each run):",
        default=str(config.system.seed) if config.system.seed is not None else "",
        validate=_validate_seed,
    )
    if seed_text is None:
        return config

    log_level = prompts.select(
        ctx,
        "Log level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=config.system.log_level,
    )
    if log_level is None:
        return config

    checkpoint_enabled = prompts.confirm(
        ctx,
        "Enable checkpointing? (saves periodic snapshots to disk)",
        default=config.storage.checkpoint_enabled,
    )
    if checkpoint_enabled is None:
        return config

    seed = int(seed_text) if seed_text.strip() else None

    return _try_merge(
        config,
        {
            "system": {
                "seed": seed,
                "log_level": log_level,
            },
            "storage": {
                "checkpoint_enabled": checkpoint_enabled,
            },
        },
    )
