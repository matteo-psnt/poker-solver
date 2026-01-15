"""Abstraction quality inspection flow for combo abstraction CLI."""

from src.interfaces.cli.flows.combo_precompute.common import _select_abstraction
from src.interfaces.cli.ui.context import CliContext


def handle_combo_quality(ctx: CliContext) -> None:
    """Show abstraction quality metrics (computed exactly at precompute time)."""
    print()
    print("=" * 60)
    print("  ABSTRACTION QUALITY")
    print("=" * 60)

    abstraction_path, metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    streets = (metadata or {}).get("streets")
    if not streets:
        print("\nNo quality statistics in metadata (regenerate the abstraction).")
        return

    print(f"\n📁 {abstraction_path.name}\n")
    header = (
        f"{'street':7s} {'buckets':>8s} {'boards':>8s} {'combos':>12s} "
        f"{'eq std':>8s} {'in-bkt std':>10s} {'var expl':>9s}"
    )
    print(header)
    print("-" * len(header))

    for street_name in ("FLOP", "TURN", "RIVER"):
        stats = streets.get(street_name)
        if not stats:
            continue
        quality = stats.get("quality", {})
        print(
            f"{street_name:7s} {stats.get('num_buckets', 0):>8,} "
            f"{stats.get('num_boards', 0):>8,} "
            f"{quality.get('combo_count', 0):>12,} "
            f"{quality.get('equity_std', 0.0):>8.4f} "
            f"{quality.get('within_bucket_std', 0.0):>10.4f} "
            f"{quality.get('variance_explained', 0.0):>9.4f}"
        )

    print()
    print("Interpretation:")
    print("  eq std      — equity spread available on the street")
    print("  in-bkt std  — equity spread forced to share a strategy (lower is better)")
    print("  var expl    — share of equity variance the buckets preserve (1.0 = lossless)")

    print("\nBucket occupancy (combos per bucket):")
    for street_name in ("FLOP", "TURN", "RIVER"):
        stats = streets.get(street_name)
        if not stats:
            continue
        quality = stats.get("quality", {})
        print(
            f"  {street_name:6s}: min={quality.get('bucket_combos_min', 0):,} "
            f"median={quality.get('bucket_combos_median', 0):,.0f} "
            f"max={quality.get('bucket_combos_max', 0):,} "
            f"(occupied {quality.get('occupied_buckets', 0)}/{quality.get('num_buckets', 0)})"
        )
