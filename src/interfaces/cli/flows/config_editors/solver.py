"""Solver section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import try_merge
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_solver_settings(ctx: CliContext, config: Config) -> Config:
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
        "Use CFR+? (floors regrets at 0 - ~100x faster convergence)",
        default=config.solver.cfr_plus,
    )
    if cfr_plus is None:
        return config

    weighting = prompts.select(
        ctx,
        "Iteration weighting:",
        choices=[
            "linear (weights later iterations more - 2-3x speedup, recommended)",
            "dcfr (Discounted CFR - Brown & Sandholm 2019)",
            "none (uniform weighting - vanilla CFR)",
        ],
        default={
            "linear": "linear (weights later iterations more - 2-3x speedup, recommended)",
            "dcfr": "dcfr (Discounted CFR - Brown & Sandholm 2019)",
            "none": "none (uniform weighting - vanilla CFR)",
        }.get(config.solver.iteration_weighting),
    )
    if weighting is None:
        return config

    iteration_weighting = weighting.split(" ")[0]

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
            "DCFR alpha - positive-regret discount exponent (recommended: 1.5):",
            default=config.solver.dcfr_alpha,
            min_value=0.001,
        )
        if dcfr_alpha is None:
            return config

        dcfr_beta = prompts.prompt_float(
            ctx,
            "DCFR beta - negative-regret discount exponent (recommended: 0.0):",
            default=config.solver.dcfr_beta,
            min_value=0.0,
        )
        if dcfr_beta is None:
            return config

        dcfr_gamma = prompts.prompt_float(
            ctx,
            "DCFR gamma - strategy discount exponent (recommended: 2.0):",
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

    return try_merge(config, overrides)
