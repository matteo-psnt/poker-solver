"""Game section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import try_merge
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_game_settings(ctx: CliContext, config: Config) -> Config:
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

    return try_merge(
        config,
        {
            "game": {
                "starting_stack": stack,
                "small_blind": small_blind,
                "big_blind": big_blind,
            }
        },
    )
