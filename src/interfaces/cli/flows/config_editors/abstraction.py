"""Card abstraction section editor for CLI config."""

from src.interfaces.cli.flows.combo_precompute import handle_combo_precompute
from src.interfaces.cli.flows.config_helpers import list_abstraction_configs, try_merge
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.training.components import build_card_abstraction
from src.shared.config import Config


def edit_card_abstraction(ctx: CliContext, config: Config) -> Config:
    print("Card Abstraction")
    print("-" * 40)

    available_configs = list_abstraction_configs(ctx)
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

    config = try_merge(config, {"card_abstraction": {"config": config_name}})

    # Verify selected abstraction availability/hash using training resolver logic.
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
