"""System section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import try_merge
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_system_settings(ctx: CliContext, config: Config) -> Config:
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

    return try_merge(
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
