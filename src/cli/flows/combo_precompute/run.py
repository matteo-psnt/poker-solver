"""Precompute execution flow for combo abstraction CLI."""

import hashlib
import multiprocessing as mp
from pathlib import Path

from src.bucketing.config import PrecomputeConfig
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.game.state import Street


def _list_available_configs(ctx: CliContext) -> list:
    """List all available abstraction config files."""
    config_dir = ctx.base_dir / "config" / "abstraction"
    configs = []

    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            if yaml_file.name != "README.md":
                configs.append(yaml_file.stem)

    return configs


def _get_config_choice(ctx: CliContext) -> tuple:
    """Prompt user for configuration choice."""
    available_configs = _list_available_configs(ctx)

    if not available_configs:
        print("\nNo configuration files found in config/abstraction/")
        print("Please create a YAML config file first.")
        return None, None

    choices = [f"{name}.yaml" for name in available_configs]
    choice = prompts.select(
        ctx,
        "Select abstraction configuration:",
        choices=choices,
    )

    if choice is None:
        return None, None

    config_name = choice.replace(".yaml", "")

    try:
        config = PrecomputeConfig.from_yaml(config_name)
        return config_name, config
    except Exception as exc:
        print(f"\nError loading config '{config_name}': {exc}")
        return None, None


TIME_PER_ITEM_BASELINE = {
    Street.FLOP: 0.37,
    Street.TURN: 0.39,
    Street.RIVER: 0.41,
}
TIME_BASELINE_WORKERS = 12
TIME_BASELINE_SAMPLES = 1000


def _estimate_time(config: PrecomputeConfig) -> None:
    """Show time estimate for precomputation."""
    print("\nEstimating precomputation time...")

    workers = config.num_workers or mp.cpu_count()

    sample_factor = config.equity_samples / TIME_BASELINE_SAMPLES
    worker_factor = TIME_BASELINE_WORKERS / workers

    estimates = {}
    total_seconds = 0.0

    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_clusters = config.num_board_clusters[street]
        reps = config.representatives_per_cluster
        num_items = num_clusters * reps

        seconds_per_item = TIME_PER_ITEM_BASELINE[street] * sample_factor * worker_factor
        street_seconds = num_items * seconds_per_item

        estimates[street] = {
            "clusters": num_clusters,
            "reps": reps,
            "items": num_items,
            "est_minutes": street_seconds / 60,
        }
        total_seconds += street_seconds

    print("\nEstimated precomputation:")
    print("-" * 50)
    for street, est in estimates.items():
        minutes = est["est_minutes"]
        if minutes < 2:
            time_str = f"{minutes * 60:.0f}s"
        elif minutes < 60:
            time_str = f"{minutes:.1f}m"
        else:
            time_str = f"{minutes / 60:.1f}h"

        print(
            f"  {street.name:6s}: {est['items']:3d} items "
            f"({est['clusters']} clusters × {est['reps']} reps) → ~{time_str}"
        )

    print("-" * 50)
    total_minutes = total_seconds / 60
    if total_minutes < 60:
        print(f"  TOTAL: ~{total_minutes:.1f} minutes (with {workers} workers)")
    else:
        print(
            f"  TOTAL: ~{total_minutes / 60:.1f} hours ({total_minutes:.0f} min, {workers} workers)"
        )
    print()


def _get_output_path(base_dir: Path, config_name: str, config: PrecomputeConfig) -> Path:
    """Generate deterministic output path based on config."""
    base_path = base_dir / "data" / "combo_abstraction"

    config_str = (
        f"{config.num_board_clusters}{config.representatives_per_cluster}"
        f"{config.representative_selection}{config.num_buckets}"
        f"{config.equity_samples}{config.seed}"
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    dirname = (
        f"buckets-F{config.num_buckets[Street.FLOP]}T{config.num_buckets[Street.TURN]}"
        f"R{config.num_buckets[Street.RIVER]}-"
        f"C{config.num_board_clusters[Street.FLOP]}C{config.num_board_clusters[Street.TURN]}"
        f"C{config.num_board_clusters[Street.RIVER]}-"
        f"s{config.equity_samples}-{config_hash}"
    )

    return base_path / dirname


def handle_combo_precompute(ctx: CliContext) -> None:
    """Handle combo-level abstraction precomputation."""
    print()
    print("=" * 60)
    print("  COMBO-LEVEL ABSTRACTION PRECOMPUTATION")
    print("=" * 60)
    print()

    config_name, config = _get_config_choice(ctx)
    if config is None:
        print("Cancelled.")
        return

    output_path = _get_output_path(ctx.base_dir, config_name, config)

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
    print(
        f"Board Clusters: F={config.num_board_clusters[Street.FLOP]}, "
        f"T={config.num_board_clusters[Street.TURN]}, "
        f"R={config.num_board_clusters[Street.RIVER]}"
    )
    print(f"Representatives/cluster: {config.representatives_per_cluster}")
    print(f"Representative selection: {config.representative_selection}")
    print(f"Equity samples: {config.equity_samples}")
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
