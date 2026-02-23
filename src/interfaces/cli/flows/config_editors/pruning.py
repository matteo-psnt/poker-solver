"""Pruning section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import try_merge
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_pruning(ctx: CliContext, config: Config) -> Config:
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
            "Pruning threshold (actions with regret below -threshold are pruned):",
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

    return try_merge(config, overrides)
