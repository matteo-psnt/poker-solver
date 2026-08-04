"""Submitting a combo abstraction precompute to the pool.

This flow used to call ``PostflopPrecomputer.precompute_all`` directly, on the
laptop, from a menu item whose own estimator printed a figure in hours. It was
the last local-compute door in the interactive CLI.

It queues a ``PRECOMPUTE`` leg instead. The invariant that made precompute look
local-only is *computed once, never recomputed* -- not *computed on a laptop* --
and a node satisfies it while also publishing the result to the share, which is
where every other machine reads it from. That is the same leg
``poker-solver-run submit-precompute`` builds, including its refusal to
republish a name that already exists.
"""

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from src.core.game.state import Street
from src.interfaces.cli.commands.submit_precompute import (
    PRECOMPUTE_TIMEOUT,
    published_abstractions,
    target_name,
)
from src.interfaces.cli.flows.config_helpers import list_config_names
from src.interfaces.cli.flows.queueing import queue_legs
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.interfaces.cloud import spec
from src.interfaces.cloud.config import CloudConfig, CloudConfigError
from src.pipeline.abstraction.config import PrecomputeConfig

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

# A prompt default, not a measured property of the pool: it matches the current
# D16als_v6 node, and `pool_vm_size` can change under it. Whatever the user
# picks is what the leg carries, so a stale value here costs an edit, not a run.
DEFAULT_NODE_WORKERS = 16


def _get_config_choice(ctx: CliContext) -> tuple[str, PrecomputeConfig] | None:
    """Prompt for an abstraction config, returning its STEM and the loaded config.

    The stem is what travels, not ``PrecomputeConfig.config_name``. A leg carries
    the stem (``LegSpec.config``) and the node resolves the YAML from it, so
    deriving the published name from anything else risks a collision check that
    describes a different abstraction than the one the node will build.
    """
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

    stem = choice.removesuffix(".yaml")

    try:
        return stem, PrecomputeConfig.from_yaml(stem)
    except Exception as exc:  # noqa: BLE001 -- interactive flow: report and return
        print(f"\nError loading config '{stem}': {exc}")
        return None


def _estimate_time(config: PrecomputeConfig, workers: int) -> None:
    """Show the wall-clock estimate for precomputing on a node.

    ``workers`` is the count the LEG will use, not this machine's core count.
    The old version read ``mp.cpu_count()``, which described the laptop -- the
    one machine that no longer does the work.
    """
    print(f"\nEstimated precomputation on a node at {workers} workers:")

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
        print(f"  TOTAL: ~{total_minutes:.1f} minutes")
    else:
        print(f"  TOTAL: ~{total_minutes / 60:.1f} hours ({total_minutes:.0f} min)")
    print()


def handle_combo_precompute(ctx: CliContext) -> None:
    """Queue a card-abstraction precompute on the pool."""
    ui.header("Precompute Abstraction (on the pool)")

    chosen = _get_config_choice(ctx)
    if chosen is None:
        print("Cancelled.")
        return
    config_stem, config = chosen

    try:
        cloud = CloudConfig.load()
        already_published = published_abstractions(cloud)
    except (CloudConfigError, ClientAuthenticationError, HttpResponseError) as error:
        ui.error(f"Could not read the share: {error}")
        print("  If this is an auth failure, `az login` and try again.")
        ui.pause()
        return

    target = target_name(config_stem)
    if target in already_published:
        # The same refusal `submit-precompute` makes, for the same reason:
        # bucket ASSIGNMENT is not pinned by the abstraction hash, so replacing
        # a published copy silently invalidates every run trained against it.
        ui.error(f"'{target}' is already published on the share.")
        print("  Republishing would silently change bucket assignment under an")
        print("  unchanged abstraction hash, invalidating the provenance of every")
        print("  run trained against it. Use `poker-solver-run submit-precompute")
        print("  --config <name> --force` if no such run matters.")
        ui.pause()
        return

    workers = prompts.prompt_int(
        ctx,
        "Workers on the node:",
        default=DEFAULT_NODE_WORKERS,
        min_value=1,
    )
    if workers is None:
        return

    _estimate_time(config, workers)

    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Config: {config_stem}.yaml")
    print(
        f"Buckets: F={config.num_buckets[Street.FLOP]}, "
        f"T={config.num_buckets[Street.TURN]}, "
        f"R={config.num_buckets[Street.RIVER]}"
    )
    print("Coverage: all canonical boards (no clustering)")
    flop_runouts = "exact (1176)" if config.flop_runouts is None else str(config.flop_runouts)
    print(f"Flop runouts: {flop_runouts} (turn/river exact)")
    print(f"Publishes to the share as: {target}")
    print(f"Leg timeout: {PRECOMPUTE_TIMEOUT}")
    print("=" * 60)

    confirm = prompts.confirm(ctx, "Queue this precompute on the pool?", default=True)
    if not confirm:
        print("Cancelled.")
        return

    queue_legs(
        lambda snapshot: [
            spec.LegSpec(
                code_snapshot=snapshot,
                op=spec.PRECOMPUTE,
                config=config_stem,
                workers=workers,
                timeout=PRECOMPUTE_TIMEOUT,
            )
        ]
    )
    ui.pause()
