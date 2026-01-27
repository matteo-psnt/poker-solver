"""Inspection/info flows for combo abstractions."""

from questionary import Choice

from src.core.game.state import Street
from src.interfaces.cli.flows.combo_precompute.common import (
    AbstractionEntry,
    _get_config_name_from_metadata,
    _list_existing_abstractions,
)
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer


def _parse_metadata_config(metadata: dict) -> PrecomputeConfig | None:
    """Parse metadata config as PrecomputeConfig."""
    config_data = metadata.get("config")
    if not isinstance(config_data, dict):
        return None
    try:
        return PrecomputeConfig.model_validate(config_data)
    except Exception:
        return None


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

    for entry in abstractions:
        path, metadata = entry.path, entry.metadata
        print(f"📁 {path.name}")
        print("   " + "-" * 57)

        parsed_config = _parse_metadata_config(metadata)
        config_name = (
            parsed_config.config_name if parsed_config else _get_config_name_from_metadata(metadata)
        )
        print(f"   Config: {config_name}")

        if parsed_config is not None:
            print(f"   Seed: {parsed_config.seed}")
            flop_runouts = (
                "exact" if parsed_config.flop_runouts is None else str(parsed_config.flop_runouts)
            )
            print(f"   Flop runouts: {flop_runouts}")

        if "streets" in metadata:
            print("\n   Street statistics:")
            for street in ["FLOP", "TURN", "RIVER"]:
                if street in metadata["streets"]:
                    stats = metadata["streets"][street]
                    quality = stats.get("quality", {})
                    print(
                        f"     {street:6s}: {stats.get('num_buckets', '?'):3} buckets, "
                        f"{stats.get('num_boards', '?'):>7,} boards, "
                        f"{quality.get('combo_count', 0):>12,} combos, "
                        f"var expl {quality.get('variance_explained', 0.0):.4f}"
                    )

        size_mb = sum(f.stat().st_size for f in path.glob("*.npy")) / (1024 * 1024)
        if size_mb > 0:
            print(f"\n   Array size: {size_mb:.1f} MB")

        print()

    view_details = prompts.confirm(
        ctx,
        "View detailed info for a specific abstraction?",
        default=False,
    )
    if view_details:
        _show_detailed_info(ctx, abstractions)


def _show_detailed_info(ctx: CliContext, abstractions: list[AbstractionEntry]) -> None:
    """Show detailed info for a selected abstraction."""
    choices: list[Choice] = [Choice(title=entry.label, value=entry) for entry in abstractions]
    choices.append(Choice(title="Back", value=None))

    selected = prompts.select(
        ctx,
        "Select abstraction for detailed view:",
        choices=choices,
    )
    if selected is not None:
        path = selected.path

        print("\n" + "=" * 60)
        print(f"DETAILED INFO: {path.name}")
        print("=" * 60)

        try:
            abstraction = PostflopPrecomputer.load(path)

            for street in [Street.FLOP, Street.TURN, Street.RIVER]:
                print(f"\n{street.name} Bucket Distribution:")
                print("-" * 60)

                bucket_counts = abstraction.get_bucket_distribution(street)
                if not bucket_counts:
                    continue

                num_unique = len(bucket_counts)
                total_combos = sum(bucket_counts.values())

                print(f"  Total combos: {total_combos:,}")
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
                    bar = "█" * max(1, round(40 * count / max_count))
                    print(f"    {bucket_id:3d}: {bar} {count:,}")

        except Exception as exc:
            print(f"\n✗ Error loading abstraction: {exc}")
