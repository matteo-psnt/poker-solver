"""Precompute execution flow for combo abstraction CLI."""

import multiprocessing as mp

from src.core.game.state import Street
from src.interfaces.cli.flows.config_helpers import list_config_names
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer


def _get_config_choice(ctx: CliContext) -> PrecomputeConfig | None:
    """Prompt user for configuration choice."""
    available_configs = list_config_names(ctx.config_dir / "abstraction")

    if not available_configs:
        print("\nNo configuration files found in config/abstraction/")
        print("Please create a YAML config file first.")
        return None

    choices = [f"{name}.yaml" for name in available_configs]
    choice = prompts.select(
        ctx,
        "Select abstraction configuration:",
        choices=choices,
    )

    if choice is None:
        return None

    config_name = choice.replace(".yaml", "")

    try:
        config = PrecomputeConfig.from_yaml(config_name)
        return config
    except Exception as exc:
        print(f"\nError loading config '{config_name}': {exc}")
        return None


# Measured single-core seconds per canonical board with the exact
# range-vs-range engine (flop scales linearly with enumerated runouts),
# plus per-street constants for canonical board enumeration.
TIME_PER_BOARD_BASELINE = {
    Street.FLOP: 1.1,
    Street.TURN: 0.05,
    Street.RIVER: 0.012,
}
BOARD_ENUMERATION_SECONDS = {
    Street.FLOP: 1.0,
    Street.TURN: 8.0,
    Street.RIVER: 55.0,
}
CANONICAL_BOARD_COUNTS = {
    Street.FLOP: 1755,
    Street.TURN: 16432,
    Street.RIVER: 134459,
}
FLOP_TOTAL_RUNOUTS = 1176


def _estimate_time(config: PrecomputeConfig) -> None:
    """Show time estimate for precomputation."""
    print("\nEstimating precomputation time...")

    workers = config.num_workers or mp.cpu_count()

    flop_runout_factor = (config.flop_runouts or FLOP_TOTAL_RUNOUTS) / FLOP_TOTAL_RUNOUTS

    estimates = {}
    total_seconds = 0.0

    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_boards = CANONICAL_BOARD_COUNTS[street]

        seconds_per_board = TIME_PER_BOARD_BASELINE[street] / workers
        if street == Street.FLOP:
            seconds_per_board *= flop_runout_factor
        street_seconds = BOARD_ENUMERATION_SECONDS[street] + num_boards * seconds_per_board

        estimates[street] = {
            "boards": num_boards,
            "est_minutes": street_seconds / 60,
        }
        total_seconds += street_seconds

    print("\nEstimated precomputation (full coverage):")
    print("-" * 50)
    for street, est in estimates.items():
        minutes = est["est_minutes"]
        if minutes < 2:
            time_str = f"{minutes * 60:.0f}s"
        elif minutes < 60:
            time_str = f"{minutes:.1f}m"
        else:
            time_str = f"{minutes / 60:.1f}h"

        print(f"  {street.name:6s}: {est['boards']:6d} canonical boards → ~{time_str}")

    print("-" * 50)
    total_minutes = total_seconds / 60
    if total_minutes < 60:
        print(f"  TOTAL: ~{total_minutes:.1f} minutes (with {workers} workers)")
    else:
        print(
            f"  TOTAL: ~{total_minutes / 60:.1f} hours ({total_minutes:.0f} min, {workers} workers)"
        )
    print()


def handle_combo_precompute(ctx: CliContext) -> None:
    """Handle combo-level abstraction precomputation."""
    print()
    print("=" * 60)
    print("  COMBO-LEVEL ABSTRACTION PRECOMPUTATION")
    print("=" * 60)
    print()

    config = _get_config_choice(ctx)
    if config is None:
        print("Cancelled.")
        return
    config_name = config.config_name or "unknown"

    output_path = abstraction_output_path(ctx.base_dir, config)

    if output_path.exists() and (output_path / "combo_abstraction.pkl").exists():
        print(f"\n[!] Abstraction already exists: {output_path}")
        print("\nThis configuration has already been precomputed.")

        overwrite = prompts.confirm(
            ctx,
            "Do you want to recompute anyway (this will overwrite)?",
            default=False,
        )

        if not overwrite:
            print("\nSkipping recomputation. Using existing abstraction.")
            print(f"Location: {output_path}")
            return

    _estimate_time(config)

    streets = [Street.FLOP, Street.TURN, Street.RIVER]

    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Config: {config_name}.yaml")
    print(f"Streets: {[s.name for s in streets]}")
    print(
        f"Buckets: F={config.num_buckets[Street.FLOP]}, "
        f"T={config.num_buckets[Street.TURN]}, "
        f"R={config.num_buckets[Street.RIVER]}"
    )
    print("Coverage: all canonical boards (no clustering)")
    flop_runouts = "exact (1176)" if config.flop_runouts is None else str(config.flop_runouts)
    print(f"Flop runouts: {flop_runouts} (turn/river exact)")
    print(f"Workers: {config.num_workers or mp.cpu_count()}")
    print(f"Output: {output_path}")
    print("=" * 60)

    confirm = prompts.confirm(
        ctx,
        "Start precomputation?",
        default=True,
    )
    if not confirm:
        print("Cancelled.")
        return

    print("\nStarting precomputation...")
    print("This may take a while. Progress will be shown.\n")

    try:
        precomputer = PostflopPrecomputer(config)
        abstraction = precomputer.precompute_all(streets=streets)
        precomputer.save(output_path)

        print("\n" + "=" * 60)
        print("PRECOMPUTATION COMPLETE!")
        print("=" * 60)
        print(f"Output saved to: {output_path}")
        print(f"Config: {config_name}")

        print("\nBucket summary:")
        for street in streets:
            num_buckets = abstraction.num_buckets(street)
            print(f"  {street.name}: {num_buckets} buckets")

    except KeyboardInterrupt:
        print("\n\nPrecomputation interrupted by user.")
    except Exception as exc:
        print(f"\nError during precomputation: {exc}")
        raise
