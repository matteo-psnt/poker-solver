"""Inspection/info flows for combo abstractions."""

from collections import Counter

from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.cli.flows.combo_precompute.common import (
    _get_config_name_from_metadata,
    _list_existing_abstractions,
)
from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.game.state import Street


def handle_combo_info(ctx: CliContext) -> None:
    """Show detailed information about existing combo abstractions."""
    print()
    print("=" * 60)
    print("  COMBO ABSTRACTION INFO")
    print("=" * 60)

    base_path = ctx.base_dir / "data" / "combo_abstraction"
    if not base_path.exists():
        print("\nNo combo abstractions found.")
        print("Run 'Precompute Combo Abstraction' to create one.")
        return

    abstractions = _list_existing_abstractions(base_path)
    if not abstractions:
        print("\nNo combo abstractions found.")
        return

    print(f"\nFound {len(abstractions)} combo abstraction(s):\n")

    for path, metadata in abstractions:
        print(f"📁 {path.name}")
        print("   " + "-" * 57)

        config_name = _get_config_name_from_metadata(metadata)
        print(f"   Config: {config_name}")

        if "config" in metadata:
            config = metadata["config"]
            print(f"   Seed: {config.get('seed', 'N/A')}")
            print(f"   Equity samples: {config.get('equity_samples', 'N/A')}")
            print(
                f"   Representatives per cluster: {config.get('representatives_per_cluster', 'N/A')}"
            )
            print(f"   Representative selection: {config.get('representative_selection', 'N/A')}")

        if "statistics" in metadata:
            print("\n   Street statistics:")
            for street in ["FLOP", "TURN", "RIVER"]:
                if street in metadata["statistics"]:
                    stats = metadata["statistics"][street]
                    num_clusters = stats.get("num_clusters", "?")
                    num_combos = stats.get("num_combos", "?")
                    num_buckets = stats.get("num_buckets", "?")
                    print(
                        f"     {street:6s}: {num_buckets:3} buckets, "
                        f"{num_clusters:3} clusters, {num_combos:6,} combos"
                    )

        pkl_file = path / "combo_abstraction.pkl"
        if pkl_file.exists():
            size_mb = pkl_file.stat().st_size / (1024 * 1024)
            print(f"\n   File size: {size_mb:.1f} MB")

        print()

    view_details = prompts.confirm(
        ctx,
        "View detailed info for a specific abstraction?",
        default=False,
    )
    if view_details:
        _show_detailed_info(ctx, abstractions)


def _show_detailed_info(ctx: CliContext, abstractions: list) -> None:
    """Show detailed info for a selected abstraction."""
    choices = []
    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        choices.append(f"{path.name} ({config_name})")

    choices.append("Back")

    choice = prompts.select(
        ctx,
        "Select abstraction for detailed view:",
        choices=choices,
    )
    if choice is None or choice == "Back":
        return

    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        if f"{path.name} ({config_name})" != choice:
            continue

        print("\n" + "=" * 60)
        print(f"DETAILED INFO: {path.name}")
        print("=" * 60)

        try:
            abstraction = PostflopPrecomputer.load(path)

            for street in [Street.FLOP, Street.TURN, Street.RIVER]:
                if street not in abstraction._buckets:
                    continue

                print(f"\n{street.name} Bucket Distribution:")
                print("-" * 60)

                all_buckets: list[int] = []
                for cluster_buckets in abstraction._buckets[street].values():
                    all_buckets.extend(cluster_buckets.values())

                if not all_buckets:
                    continue

                bucket_counts = Counter(all_buckets)
                num_unique = len(bucket_counts)

                print(f"  Total combos: {len(all_buckets):,}")
                print(f"  Unique buckets: {num_unique}")

                min_count = min(bucket_counts.values())
                max_count = max(bucket_counts.values())
                avg_count = sum(bucket_counts.values()) / len(bucket_counts)

                print(f"  Min combos per bucket: {min_count}")
                print(f"  Max combos per bucket: {max_count}")
                print(f"  Avg combos per bucket: {avg_count:.1f}")

                print("\n  Histogram (bucket → count):")
                for bucket_id in sorted(bucket_counts.keys()):
                    count = bucket_counts[bucket_id]
                    bar = "█" * (count // 50)
                    print(f"    {bucket_id:3d}: {bar} {count}")

        except Exception as exc:
            print(f"\n✗ Error loading abstraction: {exc}")

        break
