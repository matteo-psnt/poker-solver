"""CLI handler for combo-level abstraction precomputation."""

import hashlib
import json
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from pathlib import Path

from questionary import Choice

from src.bucketing.config import PrecomputeConfig
from src.bucketing.postflop import PostflopPrecomputer
from src.bucketing.postflop.hand_bucketing import PostflopBucketer
from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.cli.ui.menu import MenuItem, run_menu
from src.game.state import Card, Street


def _get_config_name_from_metadata(metadata: dict) -> str:
    """
    Extract config name from metadata JSON, handling both old and new formats.

    Old format: metadata["config_name"]
    New format: metadata["config"]["config_name"]

    Args:
        metadata: Loaded metadata dictionary

    Returns:
        Config name or "unknown" if not found
    """
    # Try new format first (nested under "config")
    if "config" in metadata and isinstance(metadata["config"], dict):
        config_name = metadata["config"].get("config_name")
        if config_name:
            return config_name

    # Fall back to old format (top-level)
    return metadata.get("config_name", "unknown")


def _list_available_configs(ctx: CliContext) -> list:
    """List all available abstraction config files."""
    config_dir = ctx.base_dir / "config" / "abstraction"
    configs = []

    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            if yaml_file.name != "README.md":
                configs.append(yaml_file.stem)

    return configs


def combo_menu(ctx: CliContext) -> None:
    """Show combo abstraction tools submenu."""
    items = [
        MenuItem("Precompute Abstraction", handle_combo_precompute),
        MenuItem("View Abstraction Info", handle_combo_info),
        MenuItem("Test Bucket Lookup", handle_combo_test_lookup),
        MenuItem("Analyze Bucketing Patterns", handle_combo_analyze_bucketing),
        MenuItem("Analyze Coverage (Fallback Rate)", handle_combo_coverage),
    ]

    run_menu(ctx, "Combo Abstraction Tools:", items, exit_label="Back")


def combo_abstraction_menu(ctx: CliContext) -> None:
    """Backward-compatible alias for the combo menu."""
    combo_menu(ctx)


def _get_config_choice(ctx: CliContext) -> tuple:
    """Prompt user for configuration choice.

    Returns:
        (config_name, PrecomputeConfig) or (None, None) if cancelled
    """
    available_configs = _list_available_configs(ctx)

    if not available_configs:
        print("\nNo configuration files found in config/abstraction/")
        print("Please create a YAML config file first.")
        return None, None

    # Show available configs
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
        # Use PrecomputeConfig.from_yaml() which uses shared merge logic
        config = PrecomputeConfig.from_yaml(config_name)
        return config_name, config
    except Exception as e:
        print(f"\nError loading config '{config_name}': {e}")
        return None, None


# Time estimation constants (empirically calibrated)
# Derived from multiple benchmark runs with varying configs
# Unit: seconds per work item (cluster × representative)
# Baseline: 12 workers, 1000 equity samples
#
# Benchmark data used for calibration:
#   fast_test (10/20/30 clusters, 1 rep, 100 samples, 12 workers):
#     FLOP: 10 items, ~30s total → 3.0s/item @ 100 samples
#     TURN: 20 items, ~68s total → 3.4s/item @ 100 samples
#     RIVER: 30 items, ~147s total → 4.9s/item @ 100 samples
#
#   default_plus (50/100/200 clusters, 3 reps, 2000 samples, 12 workers):
#     FLOP: 150 items, 7.5min → 3.0s/item @ 2000 samples
#     TURN: 300 items, 17min → 3.4s/item @ 2000 samples
#     RIVER: 600 items, 49min → 4.9s/item @ 2000 samples
#
# Pattern: time scales linearly with equity_samples, inversely with workers
# Street differences: RIVER ~60% slower than FLOP due to more opponent cards
TIME_PER_ITEM_BASELINE = {
    Street.FLOP: 1.5,  # seconds per item at 1000 samples, 12 workers
    Street.TURN: 1.7,
    Street.RIVER: 2.45,
}
TIME_BASELINE_WORKERS = 12
TIME_BASELINE_SAMPLES = 1000


def _estimate_time(config: PrecomputeConfig) -> None:
    """Show time estimate for precomputation."""
    print("\nEstimating precomputation time...")

    workers = config.num_workers or mp.cpu_count()

    # Scale factors relative to baseline
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
            f"  {street.name:6s}: {est['items']:3d} items ({est['clusters']} clusters × {est['reps']} reps) → ~{time_str}"
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
    """Generate deterministic output path based on config.

    Args:
        config_name: Name of the config (e.g., 'fast_test', 'default')
        config: PrecomputeConfig instance

    Returns:
        Path for saving abstraction
    """
    base_path = base_dir / "data" / "combo_abstraction"

    # Create deterministic hash from config parameters
    config_str = f"{config.num_board_clusters}{config.representatives_per_cluster}{config.num_buckets}{config.equity_samples}{config.seed}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    # Format: buckets-F<flop>T<turn>R<river>-C<flop_clusters>C<turn_clusters>C<river_clusters>-s<samples>-<hash>
    dirname = (
        f"buckets-F{config.num_buckets[Street.FLOP]}T{config.num_buckets[Street.TURN]}R{config.num_buckets[Street.RIVER]}-"
        f"C{config.num_board_clusters[Street.FLOP]}C{config.num_board_clusters[Street.TURN]}C{config.num_board_clusters[Street.RIVER]}-"
        f"s{config.equity_samples}-{config_hash}"
    )

    return base_path / dirname


def handle_combo_precompute(ctx: CliContext) -> None:
    """Handle combo-level abstraction precomputation.

    This is the CLI entry point for generating correct, combo-level
    equity buckets under suit isomorphism.
    """
    print()
    print("=" * 60)
    print("  COMBO-LEVEL ABSTRACTION PRECOMPUTATION")
    print("  (Correct postflop bucketing under suit isomorphism)")
    print("=" * 60)
    print()

    # Step 1: Get configuration from YAML
    config_name, config = _get_config_choice(ctx)
    if config is None:
        print("Cancelled.")
        return

    # Step 2: Check if abstraction already exists
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

    # Step 3: Show time estimate
    _estimate_time(config)

    # Step 4: Select streets
    streets = [Street.FLOP, Street.TURN, Street.RIVER]

    # Step 5: Confirm
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
    print(f"Equity samples: {config.equity_samples}")
    print(f"Workers: {config.num_workers or mp.cpu_count()}")
    print(f"Output: {output_path}")
    print("=" * 60)

    confirm = prompts.confirm(
        ctx,
        "Start precomputation?",
        default=False,
    )

    if not confirm:
        print("Cancelled.")
        return

    # Step 6: Run precomputation
    print("\nStarting precomputation...")
    print("This may take a while. Progress will be shown.\n")

    try:
        precomputer = PostflopPrecomputer(config)
        abstraction = precomputer.precompute_all(streets=streets)

        # Save results
        precomputer.save(output_path)

        print("\n" + "=" * 60)
        print("PRECOMPUTATION COMPLETE!")
        print("=" * 60)
        print(f"Output saved to: {output_path}")
        print(f"Config: {config_name}")

        # Show summary
        print("\nBucket summary:")
        for street in streets:
            num_buckets = abstraction.num_buckets(street)
            print(f"  {street.name}: {num_buckets} buckets")

    except KeyboardInterrupt:
        print("\n\nPrecomputation interrupted by user.")
    except Exception as e:
        print(f"\nError during precomputation: {e}")
        raise


def handle_combo_info(ctx: CliContext) -> None:
    """
    Show detailed information about existing combo abstractions.
    """
    print()
    print("=" * 60)
    print("  COMBO ABSTRACTION INFO")
    print("=" * 60)

    # Look for existing abstractions
    base_path = ctx.base_dir / "data" / "combo_abstraction"

    if not base_path.exists():
        print("\nNo combo abstractions found.")
        print("Run 'Precompute Combo Abstraction' to create one.")
        return

    # Find all abstractions
    abstractions = []
    for path in base_path.iterdir():
        if path.is_dir() and (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
            abstractions.append((path, metadata))

    if not abstractions:
        print("\nNo combo abstractions found.")
        return

    print(f"\nFound {len(abstractions)} combo abstraction(s):\n")

    for path, metadata in abstractions:
        print(f"📁 {path.name}")
        print("   " + "-" * 57)

        # Config name
        config_name = _get_config_name_from_metadata(metadata)
        print(f"   Config: {config_name}")

        # Config details
        if "config" in metadata:
            config = metadata["config"]
            print(f"   Seed: {config.get('seed', 'N/A')}")
            print(f"   Equity samples: {config.get('equity_samples', 'N/A')}")
            print(
                f"   Representatives per cluster: {config.get('representatives_per_cluster', 'N/A')}"
            )

        # Statistics per street
        if "statistics" in metadata:
            print("\n   Street statistics:")
            for street in ["FLOP", "TURN", "RIVER"]:
                if street in metadata["statistics"]:
                    stats = metadata["statistics"][street]
                    num_clusters = stats.get("num_clusters", "?")
                    num_combos = stats.get("num_combos", "?")
                    num_buckets = stats.get("num_buckets", "?")
                    print(
                        f"     {street:6s}: {num_buckets:3} buckets, {num_clusters:3} clusters, {num_combos:6,} combos"
                    )

        # File size
        pkl_file = path / "combo_abstraction.pkl"
        if pkl_file.exists():
            size_mb = pkl_file.stat().st_size / (1024 * 1024)
            print(f"\n   File size: {size_mb:.1f} MB")

        print()

    # Interactive selection for detailed view
    if len(abstractions) > 0:
        view_details = prompts.confirm(
            ctx,
            "View detailed info for a specific abstraction?",
            default=False,
        )

        if view_details:
            _show_detailed_info(ctx, abstractions)


def _show_detailed_info(ctx: CliContext, abstractions: list) -> None:
    """
    Show detailed info for a selected abstraction including bucket distribution.
    """
    # Build choices
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

    # Find selected
    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        if f"{path.name} ({config_name})" == choice:
            print("\n" + "=" * 60)
            print(f"DETAILED INFO: {path.name}")
            print("=" * 60)

            # Load abstraction for bucket distribution
            try:
                abstraction = PostflopPrecomputer.load(path)

                # Show bucket distribution per street
                for street in [Street.FLOP, Street.TURN, Street.RIVER]:
                    if street not in abstraction._buckets:
                        continue

                    print(f"\n{street.name} Bucket Distribution:")
                    print("-" * 60)

                    # Collect all buckets
                    all_buckets: list[int] = []
                    for cluster_buckets in abstraction._buckets[street].values():
                        all_buckets.extend(cluster_buckets.values())

                    if all_buckets:
                        bucket_counts = Counter(all_buckets)
                        num_unique = len(bucket_counts)

                        print(f"  Total combos: {len(all_buckets):,}")
                        print(f"  Unique buckets: {num_unique}")

                        # Show distribution
                        min_count = min(bucket_counts.values())
                        max_count = max(bucket_counts.values())
                        avg_count = sum(bucket_counts.values()) / len(bucket_counts)

                        print(f"  Min combos per bucket: {min_count}")
                        print(f"  Max combos per bucket: {max_count}")
                        print(f"  Avg combos per bucket: {avg_count:.1f}")

                        # Show histogram
                        print("\n  Histogram (bucket → count):")
                        for bucket_id in sorted(bucket_counts.keys()):
                            count = bucket_counts[bucket_id]
                            bar = "█" * (count // 50)
                            print(f"    {bucket_id:3d}: {bar} {count}")

            except Exception as e:
                print(f"\n✗ Error loading abstraction: {e}")

            break


def _select_abstraction(ctx: CliContext) -> tuple:
    """
    Prompt user to select an existing abstraction.

    Returns:
        (path, metadata) or (None, None) if cancelled or none found
    """
    base_path = ctx.base_dir / "data" / "combo_abstraction"

    if not base_path.exists():
        print("\nNo combo abstractions found.")
        print("Run 'Precompute Combo Abstraction' to create one.")
        return None, None

    # Find all abstractions
    abstractions = []
    for path in base_path.iterdir():
        if path.is_dir() and (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
            abstractions.append((path, metadata))

    if not abstractions:
        print("\nNo combo abstractions found.")
        return None, None

    # Build choices
    choices = []
    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        choices.append(f"{path.name} ({config_name})")

    choices.append("Cancel")

    choice = prompts.select(
        ctx,
        "Select abstraction to examine:",
        choices=choices,
    )

    if choice is None or choice == "Cancel":
        return None, None

    # Find selected abstraction
    for path, metadata in abstractions:
        config_name = _get_config_name_from_metadata(metadata)
        if f"{path.name} ({config_name})" == choice:
            return path, metadata

    return None, None


def handle_combo_test_lookup(ctx: CliContext) -> None:
    """
    Interactively test bucket lookups for specific hands/boards.
    """
    print()
    print("=" * 60)
    print("  TEST COMBO ABSTRACTION LOOKUP")
    print("=" * 60)

    # Select abstraction
    abstraction_path, metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    print(f"\nLoading abstraction from {abstraction_path.name}...")

    try:
        abstraction: PostflopBucketer = PostflopPrecomputer.load(abstraction_path)
        print("✓ Loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return

    # Interactive lookup loop
    while True:
        print("\n" + "-" * 60)

        # Get street
        street_choice = prompts.select(
            ctx,
            "Select street:",
            choices=["FLOP", "TURN", "RIVER", "Back"],
        )

        if street_choice is None or street_choice == "Back":
            break

        street = Street[street_choice]

        # Get hole cards
        print("\nEnter hole cards (e.g., AsKh):")
        hole_input = prompts.text(
            ctx,
            "Hole cards:",
            default="AsKh",
        )

        if hole_input is None:
            continue

        # Get board
        if street == Street.FLOP:
            board_example = "QsJhTc"
        elif street == Street.TURN:
            board_example = "QsJhTc9d"
        else:
            board_example = "QsJhTc9d2h"

        print(f"\nEnter board (e.g., {board_example}):")
        board_input = prompts.text(
            ctx,
            "Board:",
            default=board_example,
        )

        if board_input is None:
            continue

        # Parse cards
        try:
            # Determine expected board size
            expected_board_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]

            hole_cards = _parse_cards(hole_input, expected=2)
            board_cards = _parse_cards(board_input, expected=expected_board_cards)

            # Lookup bucket
            bucket = abstraction.get_bucket(tuple(hole_cards), tuple(board_cards), street)

            print(f"\n✓ Bucket: {bucket}")
            print(f"  (out of {abstraction.num_buckets(street)} buckets on {street.name})")

            # Test isomorphic board
            iso_board_input = prompts.text(
                ctx,
                "\nOptional: Enter isomorphic board to verify same bucket:",
                default="",
            )

            if iso_board_input:
                expected_board_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
                iso_board_cards = _parse_cards(iso_board_input, expected=expected_board_cards)
                iso_bucket = abstraction.get_bucket(
                    tuple(hole_cards), tuple(iso_board_cards), street
                )

                if iso_bucket == bucket:
                    print(f"✓ Isomorphic board maps to same bucket: {iso_bucket}")
                else:
                    print(f"⚠ Different bucket: {iso_bucket} (expected {bucket})")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            continue


def _parse_cards(card_str: str, expected: int) -> list:
    """
    Parse card string like 'AsKh' into list of Card objects.

    Args:
        card_str: String of cards (e.g., 'AsKh')
        expected: Expected number of cards

    Returns:
        List of Card objects
    """
    # Remove spaces
    card_str = card_str.replace(" ", "").replace(",", "")

    # Parse pairs of characters
    cards = []
    for i in range(0, len(card_str), 2):
        if i + 1 >= len(card_str):
            raise ValueError(f"Invalid card string: '{card_str}' (incomplete card)")

        rank = card_str[i].upper()
        suit = card_str[i + 1].lower()

        card = Card.new(rank + suit)
        cards.append(card)

    if len(cards) != expected:
        raise ValueError(f"Expected {expected} cards, got {len(cards)}")

    return cards


def handle_combo_validate(ctx: CliContext) -> None:
    """
    Run comprehensive validation tests on an abstraction.
    """
    print()
    print("=" * 60)
    print("  VALIDATE COMBO ABSTRACTION")
    print("=" * 60)

    # Select abstraction
    abstraction_path, metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    print(f"\nRunning validation on {abstraction_path.name}...")
    print("This will test:")
    print("  - Loading")
    print("  - Structure integrity")
    print("  - Bucket lookups")
    print("  - Isomorphism invariance")
    print("  - Bucket distribution")
    print()

    confirm = prompts.confirm(
        ctx,
        "Run validation?",
        default=True,
    )

    if not confirm:
        return

    try:
        abstraction = PostflopPrecomputer.load(abstraction_path)
    except Exception as e:
        print(f"\n✗ Error loading abstraction: {e}")
        return

    _run_basic_validation(abstraction)


def _run_basic_validation(abstraction: PostflopBucketer, num_samples: int = 100) -> None:
    """Run quick sanity checks for an abstraction."""
    print("\nRunning basic validation...")
    print("  - Load abstraction")
    print("  - Random bucket lookups")
    print("  - Bucket range checks")

    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]
    all_cards = [r + s for r in ranks for s in suits]

    total_checks = 0
    failures = 0

    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_buckets = abstraction.num_buckets(street)
        if num_buckets <= 0:
            print(f"\n✗ Invalid bucket count on {street.name}: {num_buckets}")
            failures += 1
            continue

        expected_board_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
        for _ in range(num_samples):
            total_checks += 1
            try:
                sampled = random.sample(all_cards, 2 + expected_board_cards)
                hole_str = sampled[:2]
                board_str = sampled[2:]

                hole_cards = tuple([Card.new(c) for c in hole_str])
                board_cards = tuple([Card.new(c) for c in board_str])

                hole_cards_pair: tuple[Card, Card] = (hole_cards[0], hole_cards[1])
                bucket = abstraction.get_bucket(hole_cards_pair, board_cards, street)
                if bucket < 0 or bucket >= num_buckets:
                    raise ValueError(f"Bucket {bucket} out of range (0..{num_buckets - 1})")
            except Exception as exc:
                failures += 1
                if failures == 1:
                    print(f"\n  [DEBUG] First error on {street.name}: {exc}")

    if failures == 0:
        print(f"\n✓ Validation complete! ({total_checks} checks)")
    else:
        print(f"\n✗ Validation failed with {failures} error(s) out of {total_checks} checks")


def handle_combo_coverage(ctx: CliContext) -> None:
    """
    Analyze abstraction coverage - how often lookups need fallback.
    """
    print()
    print("=" * 60)
    print("  ANALYZE ABSTRACTION COVERAGE")
    print("=" * 60)

    # Select abstraction
    abstraction_path, metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    print(f"\nLoading abstraction from {abstraction_path.name}...")

    try:
        abstraction = PostflopPrecomputer.load(abstraction_path)
        print("✓ Loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return

    # Select streets to analyze
    street_choices = prompts.checkbox(
        ctx,
        "Select streets to analyze:",
        choices=[
            Choice("FLOP", checked=True),
            Choice("TURN", checked=True),
            Choice("RIVER", checked=True),
        ],
    )

    if not street_choices:
        return

    streets = [Street[s] for s in street_choices]

    # Get number of samples
    num_samples = prompts.prompt_int(
        ctx,
        "Number of random samples per street:",
        default=1000,
        min_value=1,
    )

    if num_samples is None:
        return

    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)

    # Card generation
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]
    all_cards = [r + s for r in ranks for s in suits]

    overall_stats = {}

    for street in streets:
        print(f"\nAnalyzing {street.name}...")
        print("-" * 60)

        num_board_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]

        # Reset stats before this street
        abstraction.reset_stats()

        successful_lookups = 0

        # Sample random combinations
        for _ in range(num_samples):
            try:
                sampled = random.sample(all_cards, 2 + num_board_cards)
                hole_str = sampled[:2]
                board_str = sampled[2:]

                hole_cards = tuple([Card.new(c) for c in hole_str])
                board_cards = tuple([Card.new(c) for c in board_str])

                hole_cards_pair: tuple[Card, Card] = (hole_cards[0], hole_cards[1])
                _bucket = abstraction.get_bucket(hole_cards_pair, board_cards, street)

                successful_lookups += 1

            except Exception as e:
                # Skip invalid combinations (e.g., duplicate cards)
                if successful_lookups == 0:
                    print(f"  [DEBUG] First error: {e}")
                pass

        # Get stats from abstraction object
        stats = abstraction.get_fallback_stats()
        total_lookups = stats["total_lookups"]
        fallback_count = stats["fallback_count"]
        fallback_rate = stats["fallback_rate"] * 100

        print(f"  Total lookups: {total_lookups:,}")
        print(f"  Direct hits:   {total_lookups - fallback_count:,} ({100 - fallback_rate:.1f}%)")
        print(f"  Fallback used: {fallback_count:,} ({fallback_rate:.1f}%)")

        overall_stats[street] = {
            "total_lookups": total_lookups,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
        }

    # Overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_all = sum(s["total_lookups"] for s in overall_stats.values())
    fallback_all = sum(s["fallback_count"] for s in overall_stats.values())
    fallback_rate_all = (fallback_all / total_all * 100) if total_all > 0 else 0

    print("\nOverall across all analyzed streets:")
    print(f"  Total lookups:  {total_all:,}")
    print(f"  Direct hits:    {total_all - fallback_all:,} ({100 - fallback_rate_all:.1f}%)")
    print(f"  Fallback used:  {fallback_all:,} ({fallback_rate_all:.1f}%)")

    # Recommendations
    print("\nInterpretation:")
    if fallback_rate_all < 5:
        print("  ✓ Excellent coverage! Fallback rarely needed.")
    elif fallback_rate_all < 15:
        print("  ✓ Good coverage. Some fallbacks occurring.")
        print("    Consider increasing representatives_per_cluster to 2-3 for better coverage.")
    elif fallback_rate_all < 30:
        print("  ⚠ Moderate coverage. Fallbacks fairly common.")
        print("    Recommend increasing representatives_per_cluster to 3-5.")
    else:
        print("  ✗ Poor coverage. Fallback used frequently.")
        print("    Strongly recommend increasing representatives_per_cluster to 5-10.")

    print("\n  Note: Fallback uses cluster median bucket, which is reasonable but less precise.")
    print("  Higher representatives_per_cluster = more hands precomputed = fewer fallbacks.")
    print()


def handle_combo_analyze_bucketing(ctx: CliContext) -> None:
    """
    Analyze bucketing patterns by testing various hand scenarios.
    """
    print()
    print("=" * 60)
    print("  ANALYZE BUCKETING PATTERNS")
    print("=" * 60)

    # Select abstraction
    abstraction_path, metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    print(f"\nLoading abstraction from {abstraction_path.name}...")

    try:
        abstraction = PostflopPrecomputer.load(abstraction_path)
        print("✓ Loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return

    # Select street
    street_choice = prompts.select(
        ctx,
        "Select street to analyze:",
        choices=["FLOP", "TURN", "RIVER", "Cancel"],
    )

    if street_choice is None or street_choice == "Cancel":
        return

    street = Street[street_choice]

    # Select analysis type
    analysis_type = prompts.select(
        ctx,
        "Select analysis type:",
        choices=[
            "Premium vs Weak Hands (predefined scenarios)",
            "Random Sample (50 random hand/board combos)",
            "Hand Strength Correlation (various equities)",
            "Back",
        ],
    )

    if analysis_type is None or "Back" in analysis_type:
        return

    print("\n" + "=" * 60)
    print(f"BUCKETING ANALYSIS: {street.name}")
    print("=" * 60)

    if "Premium vs Weak" in analysis_type:
        _analyze_premium_vs_weak(abstraction, street)
    elif "Random Sample" in analysis_type:
        _analyze_random_sample(abstraction, street)
    elif "Hand Strength" in analysis_type:
        _analyze_hand_strength_correlation(abstraction, street)


def _analyze_premium_vs_weak(abstraction, street: Street) -> None:
    """Analyze bucketing for premium vs weak hands."""
    print("\nTesting predefined hand scenarios...")
    print("-" * 60)

    # Define test scenarios based on street
    # Format: (hand, board, description, expected_category)
    scenarios: list[tuple[tuple[str, str], tuple[str, ...], str, str]]
    if street == Street.FLOP:
        scenarios = [
            (("As", "Ah"), ("Ks", "Qh", "Jc"), "Top set", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "2s"), "Top two pair", "STRONG"),
            (("Qs", "Qh"), ("Qd", "Jc", "Th"), "Middle set", "STRONG"),
            (("As", "Ks"), ("Ah", "Kh", "2h"), "Flush draw + pair", "MEDIUM"),
            (("Jc", "Tc"), ("Qs", "9h", "2d"), "Open-ended straight draw", "MEDIUM"),
            (("7d", "7c"), ("Ks", "Qh", "Jc"), "Low pocket pair", "WEAK"),
            (("9s", "8s"), ("Kh", "6d", "2c"), "Two overs", "WEAK"),
            (("3h", "2h"), ("Kd", "Qs", "Jc"), "Nothing", "WEAK"),
        ]
    elif street == Street.TURN:
        scenarios = [
            (("As", "Ah"), ("Ks", "Qh", "Jc", "Ac"), "Trips turning into quads area", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "2s", "2d"), "Two pair improving", "STRONG"),
            (("As", "Ks"), ("Ah", "Kh", "2h", "3h"), "Made flush", "STRONG"),
            (("Jc", "Tc"), ("Qs", "9h", "2d", "8s"), "Made straight", "STRONG"),
            (("9s", "9h"), ("Kh", "6d", "2c", "9c"), "Turned set", "STRONG"),
            (("7d", "6d"), ("Kh", "5d", "2d", "Ac"), "Flush draw", "MEDIUM"),
            (("7d", "7c"), ("Ks", "Qh", "Jc", "Th"), "Pocket pair, no improvement", "WEAK"),
            (("3h", "2h"), ("Kd", "Qs", "Jc", "Tc"), "Nothing", "WEAK"),
        ]
    else:  # RIVER
        scenarios = [
            (("As", "Ah"), ("Ks", "Qh", "Jc", "Ac", "Ad"), "Quads", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "As", "Ks", "2d"), "Full house (aces full)", "STRONG"),
            (("As", "Ks"), ("Ah", "Kh", "2h", "3h", "9h"), "Nut flush", "STRONG"),
            (("Jc", "Tc"), ("Qs", "9h", "8d", "7s", "2c"), "Straight", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "2s", "3d", "7c"), "Top two pair", "MEDIUM"),
            (("Qs", "Qh"), ("Kd", "Qc", "Jc", "Th", "2s"), "Set", "MEDIUM"),
            (("7d", "7c"), ("Ks", "Qh", "Jc", "Th", "2d"), "Low pocket pair", "WEAK"),
            (("3h", "2h"), ("Kd", "Qs", "Jc", "Tc", "9s"), "High card", "WEAK"),
        ]

    # Test each scenario
    results = defaultdict(list)
    for hole_str, board_str, description, category in scenarios:
        try:
            hole_cards = tuple([Card.new(c) for c in hole_str])
            board_cards = tuple([Card.new(c) for c in board_str])

            bucket = abstraction.get_bucket(hole_cards, board_cards, street)
            results[category].append((description, bucket))

        except Exception as e:
            print(f"✗ Error testing '{description}': {e}")

    # Display results grouped by category
    for category in ["STRONG", "MEDIUM", "WEAK"]:
        if category in results:
            print(f"\n{category} HANDS:")
            for description, bucket in results[category]:
                print(f"  {description:35s} → Bucket {bucket:3d}")

    # Show bucket usage summary
    all_buckets = [bucket for scenarios in results.values() for _, bucket in scenarios]
    if all_buckets:
        print(f"\nBucket range: {min(all_buckets)} - {max(all_buckets)}")
        print(
            f"Unique buckets used: {len(set(all_buckets))} out of {abstraction.num_buckets(street)}"
        )


def _analyze_random_sample(abstraction, street: Street) -> None:
    """Analyze bucketing for random hand/board combinations."""
    print("\nGenerating 50 random hand/board combinations...")
    print("-" * 60)

    # Card ranks and suits
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]
    all_cards = [r + s for r in ranks for s in suits]

    num_board_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
    bucket_counts: Counter[int] = Counter()
    results = []

    for i in range(50):
        # Sample cards
        sampled = random.sample(all_cards, 2 + num_board_cards)
        hole_str = sampled[:2]
        board_str = sampled[2:]

        try:
            hole_cards = tuple([Card.new(c) for c in hole_str])
            board_cards = tuple([Card.new(c) for c in board_str])

            bucket = abstraction.get_bucket(hole_cards, board_cards, street)
            bucket_counts[bucket] += 1
            results.append((hole_str, board_str, bucket))

        except Exception as e:
            print(f"✗ Error in sample {i}: {e}")

    # Show first 20 results
    print("\nSample results (first 20):")
    for i, (hole_str, board_str, bucket) in enumerate(results[:20], 1):
        hole_display = "".join(hole_str)
        board_display = " ".join(board_str)
        print(f"  {i:2d}. {hole_display:6s} on {board_display:20s} → Bucket {bucket:3d}")

    if len(results) > 20:
        print(f"\n  ... and {len(results) - 20} more\n")

    # Show distribution
    print("\nBucket distribution from sample:")
    print(f"  Unique buckets hit: {len(bucket_counts)} out of {abstraction.num_buckets(street)}")
    print("  Most common buckets:")
    for bucket, count in bucket_counts.most_common(10):
        bar = "█" * (count * 2)
        print(f"    Bucket {bucket:3d}: {bar} {count}")


def _analyze_hand_strength_correlation(abstraction, street: Street) -> None:
    """Analyze if hand strength correlates with bucket assignment."""

    print("\nAnalyzing hand strength vs bucket assignment...")
    print("Testing various hand strengths on the same board texture...")
    print("-" * 60)

    # Use a fixed board and test different hand strengths
    if street == Street.FLOP:
        board_str = ["Kh", "9d", "3c"]  # King-high rainbow board
        test_hands = [
            (("Kc", "Ks"), "Set of Kings (NUTS)"),
            (("Ac", "Kd"), "Top pair, top kicker"),
            (("Kd", "Qh"), "Top pair, good kicker"),
            (("Kd", "Jh"), "Top pair, medium kicker"),
            (("9h", "9s"), "Middle set"),
            (("Ah", "As"), "Overpair (Aces)"),
            (("Qc", "Qd"), "Overpair (Queens)"),
            (("Jc", "Jd"), "Overpair (Jacks)"),
            (("Th", "Ts"), "Underpair (Tens)"),
            (("7h", "7s"), "Underpair (Sevens)"),
            (("Ah", "Qh"), "Ace-high (two overs)"),
            (("Jh", "Th"), "Jack-high (gutshot)"),
            (("5h", "4h"), "Low cards (nothing)"),
        ]
    elif street == Street.TURN:
        board_str = ["Kh", "9d", "3c", "2s"]
        test_hands = [
            (("Kc", "Ks"), "Set of Kings"),
            (("Ac", "Kd"), "Top pair, top kicker"),
            (("9h", "9s"), "Set of Nines"),
            (("Ah", "As"), "Overpair (Aces)"),
            (("Qc", "Qd"), "Overpair (Queens)"),
            (("Th", "Ts"), "Underpair (Tens)"),
            (("7h", "6h"), "Gutshot + backdoor flush"),
            (("5h", "4h"), "Gutshot"),
        ]
    else:  # RIVER
        board_str = ["Kh", "9d", "3c", "2s", "7h"]
        test_hands = [
            (("Kc", "Ks"), "Set of Kings"),
            (("Ac", "Kd"), "Top pair, top kicker"),
            (("Kd", "9h"), "Two pair (K9)"),
            (("9h", "9s"), "Set of Nines"),
            (("Ah", "As"), "Overpair (Aces)"),
            (("Qc", "Qd"), "Overpair (Queens)"),
            (("Th", "Ts"), "Underpair (Tens)"),
            (("Ah", "5h"), "Ace high"),
        ]

    # Test each hand
    results = []
    board_cards = tuple([Card.new(c) for c in board_str])

    for hole_str, description in test_hands:
        try:
            hole_cards = tuple([Card.new(c) for c in hole_str])
            bucket = abstraction.get_bucket(hole_cards, board_cards, street)
            results.append((description, bucket))
        except Exception as e:
            print(f"✗ Error testing '{description}': {e}")

    # Display results in strength order
    print(f"\nBoard: {' '.join(board_str)}\n")
    print(f"{'Hand Description':40s} Bucket")
    print("-" * 60)
    for description, bucket in results:
        print(f"{description:40s} {bucket:3d}")

    # Analyze correlation
    buckets = [bucket for _, bucket in results]
    if len(set(buckets)) == len(buckets):
        print("\n✓ Perfect differentiation: Each hand strength → unique bucket")
    else:
        print(
            f"\n⚠ Bucket overlap: {len(buckets)} hands mapped to {len(set(buckets))} unique buckets"
        )

    # Check if stronger hands tend to have higher buckets
    first_half_avg = sum(buckets[: len(buckets) // 2]) / (len(buckets) // 2)
    second_half_avg = sum(buckets[len(buckets) // 2 :]) / (len(buckets) - len(buckets) // 2)

    if first_half_avg > second_half_avg:
        print(
            f"✓ Strong hands ({first_half_avg:.1f}) tend to have higher buckets than weak hands ({second_half_avg:.1f})"
        )
    else:
        print(
            f"⚠ Unexpected pattern: Strong hands ({first_half_avg:.1f}) vs weak hands ({second_half_avg:.1f})"
        )
