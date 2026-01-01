"""Action model and resolver section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import edit_list_of_floats, try_merge
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_action_model(ctx: CliContext, config: Config) -> Config:
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
        preflop_raises = edit_list_of_floats(
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
        flop_bets = edit_list_of_floats(
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

    return try_merge(config, merge_dict)
