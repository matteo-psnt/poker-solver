"""Configuration selection and menu orchestration for CLI."""

from collections.abc import Callable

from src.interfaces.cli.flows.config_editors import (
    edit_action_model,
    edit_card_abstraction,
    edit_game_settings,
    edit_pruning,
    edit_solver_settings,
    edit_storage_settings,
    edit_system_settings,
    edit_training_params,
)
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config
from src.shared.config_loader import load_config

CATEGORY_EDITORS: dict[str, Callable[[CliContext, Config], Config]] = {
    "Training Parameters": edit_training_params,
    "Game Settings": edit_game_settings,
    "Solver Settings": edit_solver_settings,
    "Pruning": edit_pruning,
    "Action Model & Resolver": edit_action_model,
    "Card Abstraction": edit_card_abstraction,
    "Storage Settings": edit_storage_settings,
    "System Settings": edit_system_settings,
}


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

    choices = list(CATEGORY_EDITORS.keys()) + ["Done"]

    while True:
        category = prompts.select(ctx, "What would you like to edit?", choices=choices)

        if category == "Done" or category is None:
            break

        print()
        handler = CATEGORY_EDITORS.get(category)
        if handler is not None:
            config = handler(ctx, config)
        print()

    return config
